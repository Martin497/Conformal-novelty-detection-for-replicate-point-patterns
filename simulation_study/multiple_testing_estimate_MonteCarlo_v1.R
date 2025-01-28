library("GET")
library("spatstat")
library("glue")
library("hash")
library("lhs")

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

MyFit <- function(model, data) {
  if (model == "Hardcore") {
    stop("Estimating Hardcore model not implemented.")
  } else if (model == "Strauss2" | model == "Strauss1" | model == "Strauss_Mrkvicka") {
    rr <- data.frame(r=seq(0.02, 0.04, by=0.005))
    p <- profilepl(rr, Strauss, data ~ 1, aic=TRUE)
    model_fit <- as.ppm(p)
    model_params <- c(exp(model_fit$coef[[1]]), min(exp(model_fit$coef[[2]]), 1), p$param[p$iopt,])
  } else if (model == "Poisson" | model == "Poisson_Mrkvicka") {
    model_params <- c(data$n/area(data$window))
  } else if (model == "MatClust1" | model == "MatClust2" | model == "MatClust3" | model == "MatClust_Mrkvicka") {
    model_fit <- kppm(data, clusters="MatClust")
    model_params <- c(model_fit$par[[1]], model_fit$par[[2]], model_fit$mu)
  } else if (model == "LGCP") {
    model_fit <- kppm(data, clusters="LGCP", covmodel=list(model="exponential"))
    model_params <- c(model_fit$mu, model_fit$par[[1]], model_fit$par[[2]])
  } else {
    stop("No fitting null model.")
  }
  return(model_params)
}

set.seed(23)

GET_sims <- 250
GET_type <- "erl"
folder <- "MonteCarloEstimate_02"

null_data_sims <- 1
null_data_samples <- 10
data_sims <- 2000

m <- 10
m0 <- 5

#null_model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka")
#model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka")
#null_param_list <- list(c(250, 0.6, 0.03), c(200), c(200, 0.06, 1))
#param_list <- list(c(250, 0.6, 0.03), c(200), c(200, 0.06, 1))

#null_model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka")
#model_list <- list("LGCP")
#null_param_list <- list(c(250, 0.6, 0.03), c(200), c(200, 0.06, 1))
#param_list <- list(c(5, 0.6, 0.05))

#null_model_list <- list("Poisson_Mrkvicka", "MatClust_Mrkvicka")
#model_list <- list("LGCP")
#null_param_list <- list(c(200), c(200, 0.06, 1))
#param_list <- list(c(5, 0.6, 0.05))

#null_model_list <- list("LGCP")
#model_list <- list("Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP")
#null_param_list <- list(c(5, 0.6, 0.05))
#param_list <- list(c(250, 0.6, 0.03), c(200), c(200, 0.06, 1), c(5, 0.6, 0.05))

null_model_list <- list("LGCP")
model_list <- list("Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP")
null_param_list <- list(c(5, 0.6, 0.05))
param_list <- list(c(200), c(200, 0.06, 1), c(5, 0.6, 0.05))

window <- owin(c(0, 1), c(0, 1))
number_null_models = length(null_model_list)
number_models = length(model_list)
r <- Lest(rpoispp(100, win=window), correction = "translate")[['r']]


for (idx1 in 1:number_null_models) {
  null_model <- null_model_list[idx1]
  null_params <- null_param_list[idx1]
  print(glue("Estimate: {idx1}/{number_null_models}"))
  flush.console()
  
  for (nd in 1:null_data_sims) {
    ### Simulate null data
    nullData_dict <- hash()
    for (g in 1:null_data_samples) {
      Xtrain <- MySimulate(null_model, null_params, window)
      nullData_dict[[glue("{g}")]] <- Xtrain
    }
    
    ### Estimate null distribution
    if (null_model == "Poisson" | null_model == "Poisson_Mrkvicka") {
      est_params <- matrix(nrow = null_data_samples, ncol = 1)
    } else if (null_model == "Strauss2" | null_model == "Strauss1" | null_model == "Strauss_Mrkvicka") {
      est_params <- matrix(nrow = null_data_samples, ncol = 3)
    } else if (null_model == "MatClust1" | null_model == "MatClust2" | null_model == "MatClust3" | null_model == "MatClust_Mrkvicka") {
      est_params <- matrix(nrow = null_data_samples, ncol = 3)
    } else if (null_model == "LGCP") {
      est_params <- matrix(nrow = null_data_samples, ncol = 3)
    }
    for (g in 1:null_data_samples) {
      est_params[g,] <- MyFit(null_model, nullData_dict[[glue("{g}")]])
    }
    saveRDS(est_params, glue("p_values/{folder}/{null_model}_null_data_{nd}.rds"))
    
    print(glue("Simulate calibration: {idx1}/{number_null_models}"))
    flush.console()
    ### Simulate calibration data
    simCali_dict <- hash()
    for (l in 1:data_sims) {
      for (k in 1:m) {
        simCali <- matrix(nrow = length(r), ncol = GET_sims)
        for(i in 1:GET_sims) {
          ug <- runifint(1, 1, null_data_samples)
          Xcali <- MySimulate(null_model, list(est_params[ug,]), window)
          simCali[, i] <- Lest(Xcali, correction = "translate", r = r)[['trans']] - r
        }
        simCali_dict[[glue("{nd}_{l}_{k}")]] <- simCali
      }
    }
  }
  
  for (idx2 in 1:number_models) {
    print_ <- glue("Null idx: {idx1}/{number_null_models}; Alt idx: {idx2}/{number_models}")
    print(print_)
    flush.console()
    alternative_model <- model_list[idx2]
    alternative_params <- param_list[idx2]
    name <- glue("{null_model}_{alternative_model}")
    
    simTest_dict <- hash()
    for (j in 1:data_sims){
      ### Simulate test data
      for (k in 1:m) { # True null
        if (k <= m0){
          Xtest <- MySimulate(null_model, null_params, window)
        } else { # Alternative
          Xtest <- MySimulate(alternative_model, alternative_params, window)
        }
        obs <- Lest(Xtest, correction = "translate")[['trans']] - r
        simTest_dict[[glue("{j}_{k}")]] <- obs
      }
    }
    
    ### Compute p-values
    MeasureTotal <- array(0, dim=c(null_data_sims, data_sims, m))
    for (nd in 1:null_data_sims) {
      for(j in 1:data_sims) {
        ### Simulate test data
        Measure <- matrix(nrow=m)
        for (k in 1:m) {
          cset <- curve_set(r = r, obs = simTest_dict[[glue("{j}_{k}")]], sim = simCali_dict[[glue("{nd}_{j}_{k}")]])
          res_true_null <- global_envelope_test(cset, type = GET_type)
          Measure[k] <- attr(res_true_null, "M")[1]
        }
        MeasureTotal[nd,j,] <- Measure
      }
    }

    filename_rds <- glue("p_values/{folder}/{name}.rds")
    saveRDS(MeasureTotal, filename_rds)
    
    filename_txt <- glue("p_values/{folder}/{name}.txt")
    fileConn<-file(filename_txt, open="wt")
    writeLines(c(glue("Null hypothesis: {null_model}"),
                 glue("Alternative: {alternative_model}"),
                 glue("number_test_points (m): {m}"),
                 glue("number_true_nulls (m0): {m0}"),
                 glue("number_calibration_points (n): {GET_sims}"),
                 glue("GET_type: {GET_type}"),
                 glue("data_sims: {data_sims}"),
                 glue("null_data_sims: {null_data_sims}"),
                 glue("null_data_samples: {null_data_samples}")), fileConn)
    close(fileConn)
  }
}



