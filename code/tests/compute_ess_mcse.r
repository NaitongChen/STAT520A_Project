library(mcmcse)
setwd("../../data/posterior_samples")

file_names = list.files(pattern="4_diffind2_locs.csv")
seed=5

to_store1 = data.frame(matrix(NA, nrow = length(file_names), ncol = 6))
colnames(to_store1) = c("ess first cp", "mcse first cp", 
                       "ess second cp", "mcse second cp",
                       "ess third cp", "mcse third cp")
rownames(to_store1) = c("20000c1 Gibbs", "20000c1 MWG",
                       "50c1 Gibbs", "50c1 MWG",
                       "100c2 Gibbs", "100c2 MWG",
                       "100c3 Gibbs", "100c3 MWG",
                       "60c3 Gibbs", "60c3 MWG")

burnin_gibbs = c(0,0,0,0,0,
                 0,0,0,0,0,
                 0,5000,100,100,100,
                 20,20,20,20,20,
                 4000,1000,2000,500,500)
bgs = matrix(burnin_gibbs, nrow=5, byrow=T)

burnin_mwg = c(1000,1000,1000,1000,1000, # 20000 1
               0,0,0,0,0, # 50 1 
               200000,100000,1000,1000,1000, # 100 2
               1000,1000,Inf,1000,1000, # 100 3
               250000,Inf,Inf,400000,300000) # 60 3
bms = matrix(burnin_mwg, nrow=5, byrow=T)

for (i in 1:(length(file_names)/2)) {
  dat1 = read.csv(file_names[i])
  dat2 = read.csv(file_names[i+5])
  m = dim(dat1)[2]
  dat1 = dat1[c( bgs[i, seed] :dim(dat1)[1]),]
  if (bms[i,seed] < Inf) {
    dat2 = dat2[c( bms[i,seed] :dim(dat2)[1]),] 
  }
  for (j in 1:m) {
    if (m>1){
      if (bms[i,seed] < Inf) {
        to_store1[2*i-1, 2*j - 1] = ess(dat1[,j], method='obm', size='cuberoot') # 1,3,5
        to_store1[2*i-1, 2*j] = mcse(dat1[,j], method='obm', size='cuberoot')$se # 2,4,6
      }
      to_store1[2*i, 2*j - 1] = ess(dat2[,j], method='obm', size='cuberoot')
      to_store1[2*i, 2*j] = mcse(dat2[,j], method='obm', size='cuberoot')$se 
    } else {
      if (bms[i,seed] < Inf) {
        to_store1[2*i-1, 2*j - 1] = ess(dat1, method='obm', size='cuberoot') # 1,3,5
        to_store1[2*i-1, 2*j] = mcse(dat1, method='obm', size='cuberoot')$se # 2,4,6
      }
      to_store1[2*i, 2*j - 1] = ess(dat2, method='obm', size='cuberoot')
      to_store1[2*i, 2*j] = mcse(dat2, method='obm', size='cuberoot')$se 
    }
  }
}
write.csv(to_store1, "4_ess_se.csv", row.names = TRUE)
