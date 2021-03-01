library(mcmcse)
setwd("../../data/posterior_samples")

file_names = list.files(pattern="*.csv")

frame_names = data.frame(file_names)

to_store = data.frame(matrix(NA, nrow = length(file_names)/2, ncol = 6))
colnames(to_store) = c("ess first cp", "mcse first cp", 
                       "ess second cp", "mcse second cp",
                       "ess third cp", "mcse third cp")
rownames(to_store) = c("20000c1 Gibbs", "20000c1 MWG",
                       "50c1 Gibbs", "50c1 MWG",
                       "100c2 Gibbs", "100c2 MWG",
                       "100c3 Gibbs", "100c3 MWG",
                       "60c3 Gibbs", "60c3 MWG")

for (i in 1:(length(file_names)/2)) {
  if (i %% 2 == 1){
    dat1 = read.csv(file_names[i])
    dat2 = read.csv(file_names[i+10])
    m = dim(dat1)[2]
    for (j in 1:m) {
      to_store[i, 2*j - 1] = ess(dat1[,j]) # 1,3,5
      to_store[i+1, 2*j - 1] = ess(dat2[,j])
      to_store[i, 2*j] = mcse(dat1[,j])$se # 2,4,6
      to_store[i+1, 2*j] = mcse(dat2[,j])$se
    }
  }
}