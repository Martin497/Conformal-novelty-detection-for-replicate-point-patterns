library("spatstat")
#' Simulating point patterns.
#' @param model (string) The name of the model.
#' @param data (PP) The data used for model fitting as a spatstat point pattern object.
#' @return model_params (array) The estimated model parameters.
#' @export 
MyFit <- function(model, data) {
  if (model == "Strauss") {
    rr <- data.frame(r=seq(0.02, 0.04, by=0.005))
    p <- profilepl(rr, Strauss, data ~ 1, aic=TRUE)
    model_fit <- as.ppm(p)
    model_params <- c(exp(model_fit$coef[[1]]), min(exp(model_fit$coef[[2]]), 1), p$param[p$iopt,])
  } else if (model == "Poisson") {
    model_params <- c(data$n/area(data$window))
  } else if (model == "MatClust") {
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