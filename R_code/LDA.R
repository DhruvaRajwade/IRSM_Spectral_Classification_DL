library(ChemoSpec)
library(R.utils)
library(ChemoSpecUtils)
library(mcclust)
library(mclust)
library(dendextend)
library(e1071)

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
c_res <- c_pcaSpectra(Sheesh, choice = "autoscale")
library(MASS)
#ENCODING THE LABELS
Groups <- ifelse(Sheesh$groups == "Control",1,0)

int <- c_res$x[, 1:16]
#int <- c_res$x
int

wdbc.pcst <- cbind(int,Groups)
wdbc.pcst

# Calculate N
N <- nrow(wdbc.pcst)

# Create a random number vector
rvec <- runif(N)

# Select rows from the dataframe
wdbc.pcst.train <- wdbc.pcst[rvec < 0.75,]
wdbc.pcst.test <- wdbc.pcst[rvec >= 0.75,]

# Check the number of observations
nrow(wdbc.pcst.train)
wdbc.pcst.train.df <- wdbc.pcst.train


# convert matrix to a dataframe
wdbc.pcst.train.df <- as.data.frame(wdbc.pcst.train)
wdbc.pcst.test.df <- as.data.frame(wdbc.pcst.test)

model_svm = svm(Groups~ ., data = wdbc.pcst.train.df, kernel = "linear", cost = 10, scale = FALSE)
print(model_svm)
pred <- predict(model_svm, wdbc.pcst.test.df)
pred<-round(pred,digits=0)
mean(pred==wdbc.pcst.test.df$Groups)##ACCURACY
