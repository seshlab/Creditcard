
 
# Introduction and Overview

#Creditcard Fraud Detection Project Dataset is approx. 25mb data holding various transactions happened over a period of time for various set of users.
#The main objective of this project is to generate algorithm to detect Fraudrulant transactions based on existing dataset.
#Though there are several methods to generate predictive machine learning algorithm, here in this model various methods apart from  Logistic Regression are being used, here are few - SVM, Randomforest models to generate efficient algorithm.

#The key metric used to measure would be accuracy calculation there by estimate efficiency of this algorithm.

#Creditcard Fraud Detection Project is based on Creditcard transactions file, consists of 50k transactions data along with time elapsed between transactions on same card, amount involved in transaction,class (binary value , if fraud then 0 else 1) along with 28 variables used to hold client details, however they were masked due to privacy constraints.


# Methods and Analysis

#This is regarding Creditcard Fraud Detection Project. In this Project , an algorithm has been developed to predict ratings for random sample of creditcard transactions.This algorithm provides efficiency of predicted ratings upto 99%.
#This algorithm has been analysed using different methods-
  
  
 # Dataset is split into testset and trainsets for analysis. Since there are 50k records in dataset we have split into 70% train set and 30% data for testset, for some of models below dataset has been sampled with 10000 records with 149 fraud records.

#Since algorithm needs to be trained and number of fraudulent transactions are relatively low around 500 transactions in the given dataset, we split data as 70% for training and 30% for test set.
#Initially Logistic Regression has been used , resulted in accuracy of 99.8%.
#with Decission tree model - accuracy is 99.925%
#with SVM Model- 98.8% 
#and finally, Random Forest model provides an 100% accuracy.


#tinytex used for pdf output,installing tinytex


#system("ls ../", intern=TRUE)
#if(!require(tinytex)) install.packages("tinytex")

#Dataset file needs to be downloaded and placed at project location
#Below code downloads file and reads into csv variable



read_url_csv <- function(url){
  tmpFile <- tempfile()
  download.file(url, destfile = tmpFile)
  url_csv <- readr::read_csv(tmpFile)
  return(url_csv)
}
onedrive_url <- "https://raw.githubusercontent.com/seshlab/Creditcard/master/creditcarddataset1.csv"
csv <- read_url_csv(onedrive_url)
head(csv)


#Below are required packages 
#to load libraries


library(readr)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(caTools)
library(caret)



#Below code turns file into factor format


datacard <- csv
str(datacard)
#count of rows in datacard

nrow(datacard)
#sample data from datacard
head(datacard)
# convert class variable to factor
datacard$Class <- factor(datacard$Class)


#Predictive Modelling
#since this algorithm relatively less fraudulent data, datasplit at 70:30 works well inorder to acheive well balanced data both fraud and non fraudulent transactions. 
#split data 70:30


set.seed(1)
splitdatacard <- sample.split(datacard$Class, SplitRatio = 0.7)
traindata <- subset(datacard, splitdatacard == T)
testdata <- subset(datacard, splitdatacard == F)

# check output Class distributiion
table(testdata$Class)



#baseline accuracy --> 99.826785 %

##logistic regression



glm.model <- glm(Class ~ ., data = traindata, family = "binomial")
glm.predict <- predict(glm.model, testdata, type = "response")
table(testdata$Class, glm.predict > 0.5)
fitted.results.cat <- ifelse(glm.predict > 0.5,"1","0")

fitted.results.cat<-as.factor(fitted.results.cat)
confusionMatrix(testdata$Class,fitted.results.cat)
plot(glm.model)

##99.82 % accuracy using logistic regression model.

##Decision tree model


tree.model <- rpart(Class ~ ., data = traindata, method = "class", minbucket = 20)
prp(tree.model) 
tree.predict <- predict(tree.model, testdata, type = "class")
confusionMatrix(testdata$Class, tree.predict)
#plot(tree.model)

##99.89 % accuracy (best) using decision tree.
## Noticed 0.07 % rise in accuracy so far let us analyze using only part of data that is 10000 records along with fraudulent transactions(149 records) this will result in well balanced data with 10149 records


#Now we only keep 10000 rows of data with class = 0

data_nofraud <- subset(datacard, datacard$Class == 0)
data_fraud <- subset(datacard, datacard$Class == 1)
nrow(data_nofraud)
nrow(data_fraud)
data_nofraud <- data_nofraud[1:10000, ]
nrow(data_nofraud)
data_all <- rbind(data_nofraud, data_fraud)
nrow(data_all)


##split data 70:30 , and find accuracy for different models

set.seed(1)
data_split <- sample.split(data_all$Class, SplitRatio = 0.7)
traindata <- subset(data_all, data_split == T)
testdata <- subset(data_all, data_split == F)

table(testdata$Class)

baseline accuracy --> 95.298602 %

## logistic regression

glm.model <- glm(Class ~ ., data = traindata, family = "binomial", control = list(maxit = 50))
glm.predict <- predict(glm.model, testdata, type = "response")
table(testdata$Class, glm.predict > 0.5)
fitted.results.cat <- ifelse(glm.predict > 0.5,"1","0")

fitted.results.cat<-as.factor(fitted.results.cat)
confusionMatrix(testdata$Class,fitted.results.cat)


### 99.8 % accuracy

## SVM model

svm.model <- svm(Class ~ ., data = traindata, kernel = "radial", cost = 1, gamma = 0.1)
svm.predict <- predict(svm.model, testdata)
confusionMatrix(testdata$Class, svm.predict)


### 99.21 % accuracy

## Decision Tree Model

tree.model <- rpart(Class ~ ., data = traindata, method = "class", minbucket = 20)
prp(tree.model) 
tree.predict <- predict(tree.model, testdata, type = "class")
confusionMatrix(testdata$Class, tree.predict)
#plot(tree.model)

### 99.67 % accuracy !!

## Let's try random forest as well..

## random forest model

set.seed(10)
rf.model <- randomForest(Class ~ ., data = traindata,
                         ntree = 500, nodesize = 20)

rf.predict <- predict(rf.model, testdata)
confusionMatrix(testdata$Class, rf.predict)
#minimal error noticed below
plot(rf.model)
#used to understand the volume of fraudulent transactions(value=1)
plot(rf.predict)





### Random forest gives 100 % accuracy

# Results

#As per analysis in understanding movielens algorithm , noticed that using linear regression it didnt fetch expected results due to bias values. However noticed better RMSEs with inclusion of movie bias on the base model and RMSE is around 0.986.

#However if User effect is included along with Movie effect the result here is 0.908 with regularization it is 0.965.

#With varying sample sets between 0 and 10, noticed that RMSE is improved and came down 0.864

#Here is plot of used samples Vs RMSEs
#This plot helps to identify the lowest recorded RMSE for sample of lamdas.


# Conclusion

#Brief Summary: This algorithm utilises 10492 transaction from 0.2 million records in actual dataset.
#This algorithm provides an extreme efficiency by using Randomforest model and decission tree model among other models logistic regression, SVM model etc.

#Limitations and Future work: However there are few limitations to this algorithm in terms of balanced and over sampling.
#This data can be further balanced such that fraudulant transactions ratio to non fraudulent transactions is maintained in balance.
#As we are not completely using entire dataset here , different data samples and well balanced datasets can be studied for further analysis with a close study on different fraud trends like elapsed times between transactions at different rates.


