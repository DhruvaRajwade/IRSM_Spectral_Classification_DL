library(ChemoSpec)
library(R.utils)
library(ChemoSpecUtils)
library(baseline)
library(mcclust)
library(mclust)
library(amap)
library(dendextend)
library(seriation)#HeatMAP
options(ChemoSpecGraphics = 'ggplot2')
#data<-read.csv('Hyperspec.csv')
matrix2SpectraObject(gr.crit = c("Control", "Test"),
                     gr.cols = c("auto"),
                     freq.unit = "/cm",
                     int.unit = "/cm",
                     descrip = "no description provided",
                     in.file = 'HyperspecV2.csv',sep=",",
                     out.file = "Pray",
                     chk = TRUE)
 Sheesh<- loadObject("Pray.RData")
 
 #myt <- expression(bolditalic(Serenoa)~bolditalic(repens)~bold(Extract~IR~Spectra))
 #Test<- baselineSpectra(Sheesh, int = FALSE, method = "modpolyfit", retC = TRUE)
 
#Loading Plot 
c_res <- c_pcaSpectra(Sheesh, choice = "noscale")
p <- plotScores(Sheesh, c_res, pcs = c(2,1), ellipse = "rob", tol ="none")
p <- plotLoadings(Sheesh, c_res, loads = c(2, 1), ref = 1)
p

     
#HCA

HCA<-hcaSpectra(
  Sheesh,
  c.method = "complete",
  d.method = "euclidean",
  use.sym = FALSE,
  leg.loc = FALSE
)




#PCA
c_res <- c_pcaSpectra(Sheesh)
p <- plotScores(Sheesh, c_res, pcs = c(1,2), ellipse = "rob", tol ="none",leg.loc = FALSE)
p

#Correlation Plot
myt <- expression(bolditalic(Serenoa)~bolditalic(repens)~bold(Extract~IR~Spectra))
p <- sPlotSpectra(Sheesh, c_res, pc = 1, tol = 0.005)
p <- p + ggtitle(myt)
p
hcaScores(Sheesh, c_res, scores = c(1:5), main = myt)
hmapSpectra(Sheesh,row_labels = Sheesh$names, col_labels = as.character(round(Sheesh$freq)))





#IDK
model <- mclustSpectra(Sheesh, c_res, plot = "BIC", main = myt)
model <- mclustSpectra(Sheesh, c_res, plot = "proj", main = myt)
model <- mclustSpectra(Sheesh, c_res, plot = "errors", main = myt, truth = Sheesh$groups)


  