#' @author: Martin Voigt Vejling
#' Emails: mvv@math.aau.dk
#'         mvv@es.aau.dk
#'         martin.vejling@gmail.com
#' 
#' Main script for the proposed conformal multiple Monte Carlo test (CMMCTest).
#' In this script, the conformal p-values are computed for the multiple
#' Monte Carlo testing setup.
#' 
library("GET")
library("spatstat")
library("glue")
library("hash")
source("MySimulate.R")

set.seed(35)

test_function <- "J" # Options: L, J

GET_sims <- 2500
n_list <- c(GET_sims)
GET_type <- "erl"
folder <- "CMMCTest_01"
tie_breaking_method <- "joint" # Options: parallel, joint

data_sims <- 2000

m <- 10
m0 <- 5

null_model_list <- list("Strauss", "Poisson", "LGCP")
model_list <- list("Strauss", "Poisson", "LGCP")
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


for (idx1 in 1:number_null_models) {
  null_model <- null_model_list[idx1]
  null_params <- null_param_list[idx1]

  ### Simulate calibration data
  simCali_dict <- hash()
  for(j in 1:data_sims) {
    simCali <- matrix(nrow = length(r), ncol = GET_sims)
    for(i in 1:GET_sims) {
      Xcali <- MySimulate(null_model, null_params, window)
      if (test_function == "L") {
        simCali[, i] <- Lest(Xcali, correction = "translate", r = r)[['trans']] - r
      } else if (test_function == "J") {
        simCali[, i] <- Jest(Xcali, correction = "km", r = r)[['km']]
      }
    }
    simCali[is.na(simCali)] <- 0
    simCali_dict[[glue("{j}")]] <- simCali
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
        if (test_function == "L") {
          obs <- Lest(Xtest, correction = "translate", r = r)[['trans']] - r
        } else if (test_function == "J") {
          obs <- Jest(Xtest, correction = "km", r = r)[['km']]
          obs[is.na(obs)] <- 0
        }
        simTest_dict[[glue("{j}_{k}")]] <- obs
      }
    }

    ### Compute p-values
    MeasureTotal <- array(0, dim=c(data_sims, length(n_list), m))
    for(j in 1:data_sims) {
      if (tie_breaking_method == "parallel") {
        for (k in 1:m) {
          nidx <- 1
          for (nuse in n_list) {
            cset <- curve_set(r = r, obs = simTest_dict[[glue("{j}_{k}")]], sim = simCali_dict[[glue("{j}")]][,1:nuse])
            res_true_null <- forder(cset, measure = GET_type)
            MeasureTotal[j,nidx,k] <- (1 + sum(res_true_null[2:(nuse+1)] <= res_true_null[1], na.rm=TRUE))/(nuse + 1)
            nidx <- nidx + 1
          }
        }
      } else if (tie_breaking_method == "joint") {
        nidx <- 1
        for (nuse in n_list) {
          simTest <- array(0, dim=c(length(r), m))
          for (k in 1:m) {
            simTest[,k] <- simTest_dict[[glue("{j}_{k}")]]
          }
          data <- cbind(simCali_dict[[glue("{j}")]][,1:nuse], simTest)
          cset <- curve_set(r = r, obs = data)
          res <- forder(cset, measure = GET_type)
          for (k in 1:m) {
            MeasureTotal[j,nidx,k] <- (1 + sum(res[1:nuse] <= res[nuse+k], na.rm=TRUE))/(nuse + 1)
          }
          nidx <- nidx + 1
        }
      }
    }

    filename_rds <- glue("p_values/{folder}/{name}.rds")
    saveRDS(MeasureTotal, filename_rds)
  }
}



