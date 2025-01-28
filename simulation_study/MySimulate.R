library("spatstat")
#' Simulating point patterns.
#' @param model (string) The name of the model.
#' @param params (list) The parameters of the model.
#' @param window (owin) The domain of observation as a spatstat owin object.
#' @return X (PP) The simulated point pattern as a spatstat point pattern object.
#' @export 
MySimulate <- function(model, params, window){
  if (model == "Hardcore") {
    X <- rHardcore(params[[1]][1], R=params[[1]][2], W=window)
  } else if (model == "Strauss") {
    X <- rStrauss(params[[1]][1], params[[1]][2], R=params[[1]][3], W=window)
  } else if (model == "Poisson") {
    X <- rpoispp(params[[1]], win=window)
  } else if (model == "MatClust") {
    X <- rMatClust(params[[1]][1], params[[1]][2], params[[1]][3], win=window)
  } else if (model == "LGCP") {
    X <- rLGCP(model="exponential", mu=params[[1]][1], var=params[[1]][2], scale=params[[1]][3], win=window)
  } else {
    stop("No fitting null model.")
  }
  return(X)
}