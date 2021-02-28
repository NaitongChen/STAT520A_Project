library(mcmcse)
setwd("../../data/posterior_samples")

file_names = list.files(pattern="*.csv")

frame_names = data.frame(file_names)

to_store = data.frame(matrix(NA, nrow = length(file_names), ncol = 4))

for (i in 1:length(dat_names)) {
  dat = read.csv(file_names[i])
  n = dim(dat)[1]
  p = dim(dat)[2]
  to_store[i,1] = tryCatch({
    multiESS(dat, method="bm")
  }, error = function(errorCondition){
    return(NA)
  })
  to_store[i,2] = tryCatch({
    multiESS(dat, method="obm")
  }, error = function(errorCondition){
    return(NA)
  })
  to_store[i,3] = n^p * det(cov(dat)) / (to_store[i,1]^p)
  to_store[i,4] = n^p * det(cov(dat)) / (to_store[i,2]^p)
}

dat = read.csv(file_names[4])
n = dim(dat)[1]
show = 5000
plot((n-show):n, dat[(n-show):n,1])

dat = read.csv(file_names[4])
n = dim(dat)[1]
plot(1:n, dat[1:n,1])