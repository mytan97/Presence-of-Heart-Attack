library(foreign)
install.packages('glmnet')
library(glmnet)
install.packages('ROCR')
library(ROCR)
install.packages('caret')
library(caret)
install.packages('frequency')
library(frequency) 

heart = read.csv('heart.csv', header = TRUE)
attach(heart)
head(heart)
str(heart)
summary(heart)
dim(heart)

#Check for Missing Values
missing = heart[!complete.cases(heart),] #no missing values
#Check for duplicates
heart = heart[!duplicated(heart),]
dim(heart) #one duplicate obs is deleted and sample size is 302


#Density plot of the response variable vs the 12 predictors
scales =list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x = heart[, 1:12], 
            y = as.factor(heart$target),
            plot = "density", 
            scales = scales,
            auto.key = list(columns = 2))

#Generate correlation matrix between the variables
library(corrplot)
M = cor(heart)
corrplot(M, method="number")


set.seed(123456789)
n1=302
train = which(runif(n1)<= .7)
data_train = heart[train,] #training data
dim(data_train)

data_test = heart[-train,] #test data
dim(data_test)

#get predictor matrix (design matrix)
x = model.matrix(target ~ age+as.factor(sex)+as.factor(cp)+trestbps+chol+as.factor(fbs)+as.factor(restecg)+
                   thalach+as.factor(exang)+oldpeak+as.factor(slope)+ca, heart)[,-1]

#get response vector
y=heart$target


####################################################################################################################
#Fit conventional logistic regression using glm function

data = data.frame(y,x)
dim(data)
fitdata = data[train,]

dim(fitdata)
model1 = glm(y~., data=fitdata, family="binomial")
summary(model1)

model2=step(model1) #stepwise selection
summary(model2)

coef(model2)
exp(coef(model2))
par(mfrow=c(2,2))
plot(model2)

#Homser and Lemeshow GOF test
install.packages("ResourceSelection")
library(ResourceSelection)
hoslem.test(x = fitdata$y, y = fitted(model2), g = 10) #p-value = 0.8911 (accept null, no lack of fit)


testdata = data[-train,]
dim(testdata)
pred = predict(model2, newdata=testdata, type = "response") #predict using stepwise model
ypred = ifelse(pred>.5, 1, 0)
length(which(ypred==y[-train]))/(n1-length(train)) #compute correct classification rate/test set prediction result=0.7977528


# check the area under ROC for the test set and plot
pred1 = prediction(pred, y[-train]) #transform the input data into a standardized format
performance(pred1,"auc")@y.values[[1]] #get AUC of the ROC for the test set (AUC=0.9005102)
perf = performance(pred1,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
title(main = "ROC curve- Conventional (AUC = 0.9005102)",  cex.main = 1,   font.main= 1)


###########################LASSO####################################################################33
cv.out =cv.glmnet(x[train ,],y[train], alpha =1, family = "binomial", type.measure = 'auc')
plot(cv.out)

bestlam =cv.out$lambda.min
bestlam #0.00171024

out0 = glmnet(x[train,],y[train], alpha =1, family="binomial") #automatic sequence of lambdas
out0$lambda
plot(out0, "lambda", label=T)


out = predict(out0 ,type ="coefficients", s=bestlam, exact=TRUE)
out

#prediction using lasso logistic regression
pred = predict(out0, s=bestlam ,newx=x[-train ,], type="response") #type="response" give prob
ypred = ifelse(pred>.5, 1, 0) #predict class
length(which(ypred==y[-train]))/(n1-length(train)) #compute correct classification rate/test set prediction result=0.8202247


# check the area under ROC for the test set and plot
pred1 = prediction(pred, y[-train]) #transform the input data into a standardized format
performance(pred1,"auc")@y.values[[1]] #get AUC of the ROC for the test set (AUC=0.9096939)
perf = performance(pred1,"tpr","fpr")
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
title(main = "ROC curve- LASSO (AUC = 0.9096939)",  cex.main = 1,   font.main= 1)

####################################################################################################################
#Elastic Net
data = data.frame(y,x)
dim(data)

fitdata = data[train,]
dim(fitdata)

testdata = data[-train,]
dim(testdata)

num = 10
alph = seq(0,1, length=num)
alph
MSE = rep(0, num)

set.seed(1)
for (i in 1:num){
  cv.elasticnet = cv.glmnet(x[train,], y[train], alpha=alph[i])
  bestlam =cv.elasticnet$lambda.min
  bestlam
  elasticnet = glmnet(x[train,], y[train], alpha=alph[i], lambda=bestlam)
  elas.coef = coef(elasticnet)[1:17,]
  #elas.coef[elas.coef!=0]
  elas.pred=predict(elasticnet, s=bestlam ,newx=x[-train ,])
  MSE[i] = mean(( elas.pred -y[-train])^2)
}

MSE
min(MSE) #0.1261455(best alpha has the smallest MSE) ---- alpha=0.2222222 


cv.out =cv.glmnet(x[train ,],y[train], alpha =0.2222222, family = "binomial", type.measure = 'auc')
plot(cv.out)

set.seed(123456789)
bestlam =cv.out$lambda.min
bestlam #0.05958795

out0 = glmnet(x[train,],y[train], alpha =0.2222222, family="binomial") #automatic sequence of lambdas
out0$lambda
plot(out0, "lambda", label=T)

out = predict(out0 ,type ="coefficients", s=bestlam, exact=TRUE)
out

#prediction using elastic net regression 
pred = predict(out0, s=bestlam ,newx=x[-train ,], type="response") #type="response" give prob
ypred = ifelse(pred>.5, 1, 0) #predict class
length(which(ypred==y[-train]))/(n1-length(train)) #compute correct classification rate/test set prediction result=0.8426966


# check the area under ROC for the test set and plot
pred1 = prediction(pred, y[-train]) #transform the input data into a standardized format
performance(pred1,"auc")@y.values[[1]] #get AUC of the ROC for the test set (AUC=0.9081633)
perf = performance(pred1,"tpr","fpr")
plot(perf,colorize=FALSE, col="black") # plot ROC curve
lines(c(0,1),c(0,1),col = "gray", lty = 4 )
title(main = "ROC curve- Elastic Net (AUC = 0.9081633)",  cex.main = 1,   font.main= 1)

