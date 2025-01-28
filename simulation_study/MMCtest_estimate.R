#' @author: Martin Voigt Vejling
#' Emails: mvv@math.aau.dk
#'         mvv@es.aau.dk
#'         martin.vejling@gmail.com
#' 
#' Main script for the naïve multiple Monte Carlo test (CMMCTest).
#' In this script, independent p-values are computed for the multiple
#' Monte Carlo testing setup where the null sample is augmented
#' by a parametric method.
#' 
library("GET")
library("spatstat")
library("glue")
library("hash")
library("lhs")
source("MySimulate.R")
source("MyFit.R")

set.seed(35)

GET_sims <- 250
GET_type <- "erl"
folder <- "MMCTest_estimate_01"

null_data_sims <- 1
null_data_samples <- 10
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
  }
}



