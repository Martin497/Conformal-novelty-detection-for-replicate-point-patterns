library("GET")
library("spatstat")
library("glue")
library("hash")

MySimulate <- function(model, params, window){
  if (model == "Hardcore") {
    X <- rHardcore(params[[1]][1], R=params[[1]][2], W=window)
  } else if (model == "Strauss2" | model == "Strauss1" | model == "Strauss_Mrkvicka") {
    X <- rStrauss(params[[1]][1], params[[1]][2], R=params[[1]][3], W=window)
  } else if (model == "Poisson" | model == "Poisson_Mrkvicka") {
    X <- rpoispp(params[[1]], win=window)
  } else if (model == "MatClust1" | model == "MatClust2" | model == "MatClust3" | model == "MatClust_Mrkvicka") {
    X <- rMatClust(params[[1]][1], params[[1]][2], params[[1]][3], win=window)
  } else if (model == "LGCP") {
    X <- rLGCP(model="exponential", mu=params[[1]][1], var=params[[1]][2], scale=params[[1]][3], win=window)
  } else {
    stop("No fitting null model.")
  }
  return(X)
}

set.seed(35)

test_function <- "L" # Options: L, J

GET_sims <- 250
n_list <- c(GET_sims)
GET_type <- "erl"
folder <- "GET_05"

data_sims <- 2000

m <- 10
m0 <- 5

#null_model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP")
#model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP")
#null_param_list <- list(c(250, 0.6, 0.03), c(200), c(200, 0.06, 1), c(5, 0.6, 0.05))
#param_list <- list(c(250, 0.6, 0.03), c(200), c(200, 0.06, 1), c(5, 0.6, 0.05))

null_model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP")
model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP")
null_param_list <- list(c(250, 0.6, 0.03), c(200), c(5, 0.6, 0.05))
param_list <- list(c(250, 0.6, 0.03), c(200), c(5, 0.6, 0.05))

window <- owin(c(0, 1), c(0, 1))
number_null_models = length(null_model_list)
number_models = length(model_list)
if (test_function == "L") {
  r <- Lest(rpoispp(100, win=window), correction = "translate")[['r']]
} else if (test_function == "J") {
  r <- Jest(rpoispp(100, win=window), correction = "km")[['r']]
  r <- r[1:200]
}
r_tot <- NULL
for (k in 1:m) {
  r_tot <- c(r_tot, r)
}


for (idx1 in 1:number_null_models) {
  null_model <- null_model_list[idx1]
  null_params <- null_param_list[idx1]

  ### Simulate calibration data
  simCali_dict <- hash()
  for(j in 1:data_sims) {
    simCali <- matrix(nrow = length(r_tot), ncol = GET_sims)
    for(i in 1:GET_sims) {
      simCali_stack <- NULL
      for (k in 1:m) {
        Xcali <- MySimulate(null_model, null_params, window)
        if (test_function == "L") {
          simCali_inner <- Lest(Xcali, correction = "translate", r = r)[['trans']] - r
        } else if (test_function == "J") {
          simCali_inner <- Jest(Xcali, correction = "km", r = r)[['km']]
          simCali_inner[is.na(simCali_inner)] <- 0
        }
        simCali_stack <- c(simCali_stack, simCali_inner)
      }
      simCali[, i] <- simCali_stack
    }
    simCali_dict[[glue("{j}")]] <- simCali
  }
  for (idx2 in 1:number_models) {
    print_ <- glue("Null idx: {idx1}/{number_null_models}; Alt idx: {idx2}/{number_models}")
    print(print_)
    flush.console()
    alternative_model <- model_list[idx2]
    alternative_params <- param_list[idx2]
    name <- glue("{null_model}_{alternative_model}")

    ### Simulate test data
    simTest_dict <- hash()
    for (j in 1:data_sims){
      obs_stack <- NULL
      for (k in 1:m) { # True null
        if (k <= m0){
          Xtest <- MySimulate(null_model, null_params, window)
        } else { # Alternative
          Xtest <- MySimulate(alternative_model, alternative_params, window)
        }
        if (test_function == "L") {
          obs_inner <- Lest(Xtest, correction = "translate", r = r)[['trans']] - r
        } else if (test_function == "J") {
          obs_inner <- Jest(Xtest, correction = "km", r = r)[['km']]
          obs_inner[is.na(obs_inner)] <- 0
        }
        obs_stack <- c(obs_stack, obs_inner)
      }
      simTest_dict[[glue("{j}")]] <- obs_stack
    }

    ### Compute p-values
    MeasureTotal <- array(0, dim=c(data_sims, length(n_list)))
    for(j in 1:data_sims) {
      nidx <- 1
      for (nuse in n_list) {
        cset <- curve_set(r = r_tot, obs = simTest_dict[[glue("{j}")]], sim = simCali_dict[[glue("{j}")]][,1:nuse])
        res <- forder(cset, measure = GET_type)
        MeasureTotal[j,nidx] <- (1 + sum(res[2:(nuse+1)] <= res[1], na.rm=TRUE))/(nuse + 1)
        nidx <- nidx + 1
      }
    }

    filename_csv <- glue("p_values/{folder}/{name}.csv")
    write.csv(MeasureTotal, filename_csv)
  }
}



