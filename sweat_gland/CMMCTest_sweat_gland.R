library("GET")
library("spatstat")
library("glue")
library("hash")
library("lhs")

ImportSimulation <- function(subject_id, sim, window){
  data <- read.delim(glue("simulations/subject_{subject_id}/gm_subject_{subject_id}_sim_{sim}_pythonized.txt"))[[1]]
  pp <- array(0, dim=c(length(data), 2))
  for (i in 1:length(data)) {
    line <- data[i]
    x <- as.double(strsplit(line, ",")[[1]][1])
    y <- as.double(strsplit(line, ",")[[1]][2])
    point <- c(x, y)
    pp[i,] <- point
  }
  X <- as.ppp(pp, W=owin(c(window$x0, window$x1), c(window$y0, window$y1)))
  return(X)
}

setwd("/home/martin/Documents/GitHub/ConformalTesting/sweat_gland")
set.seed(13)

control_ids <- c(96, 149, 203, 205)
MNA_diagnosed_ids <- c(23, 36, 42, 50, 73)
MNA_suspected_ids <- c(10, 20, 40, 61, 71)

train_ids <- MNA_diagnosed_ids
test_ids <- c(MNA_suspected_ids, control_ids)
#train_ids <- control_ids
#test_ids <- c(MNA_suspected_ids, MNA_diagnosed_ids)
g <- length(train_ids)
m <- length(test_ids)

GET_sims <- 5000
GET_type <- "erl"
folder <- "p_values"
name <- "gm_diagnosed_Conformal_n5000"

meta <- read.csv("data/meta.csv")
data <- read.csv("data/glands.csv")
window <- subset(meta, subjectid==20)
r <- Lest(rpoispp(4e-05, win=owin(c(window$x0, window$x1), c(window$y0, window$y1))), correction = "translate")[['r']]

### Import/simulate calibration data
simCali <- matrix(nrow = length(r), ncol = GET_sims)
sim_counter <- array(1, dim=(g))
for(i in 1:GET_sims) {
  ug <- runifint(1, 1, g)
  Xcali <- ImportSimulation(train_ids[ug], sim_counter[ug], window)
  simCali[, i] <- Lest(Xcali, correction = "translate", r = r)[['trans']] - r
  sim_counter[ug] = sim_counter[ug] + 1
}

### Import test data
simTest <- matrix(nrow = length(r), ncol = m)
for (k in 1:m) {
  xy <- subset(data, subjectid==test_ids[k])
  Xtest <- ppp(xy$x, xy$y, window=owin(c(window$x0, window$x1), c(window$y0, window$y1)))
  simTest[, k] <- Lest(Xtest, correction = "translate")[['trans']] - r
}

### Compute p-values
Measure <- matrix(nrow=m)
for (k in 1:m) {
  cset <- curve_set(r = r, obs = simTest[, k], sim = simCali)
  res_true_null <- global_envelope_test(cset, type = GET_type)
  Measure[k] <- attr(res_true_null, "M")[1]
}

filename_csv <- glue("{folder}/{name}.csv")
write.csv(Measure, filename_csv)

#filename_txt <- glue("{folder}/{name}.txt")
#fileConn<-file(filename_txt, open="wt")
#writeLines(c(glue("number_test_points (m): {m}"),
#             glue("number_calibration_points (n): {GET_sims}"),
#             glue("GET_type: {GET_type}")), fileConn)
#close(fileConn)
#write.csv(Measure, filename_csv)


