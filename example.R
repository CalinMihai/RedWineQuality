# Red Wine Values 

# Regression, numeric inputs

# Dataset Description: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009


# load libraries
library(mlbench)
library(caret)
library(corrplot)

# attach the redWine dataset
redWine = read.csv("C:/Users/mihby/Desktop/ML/winequality-red.csv")

#removing residual sugar and chlorides due to extreme outliers
redWine <- redWine[,-4]
redWine <- redWine[,-4]

#creating a vector for the new rating system
vector <- as.vector(redWine[,10], mode = "numeric")
length(vector)

#populating the vector
for (i in 1:length(vector)) {
  if(vector[i] >= 7){
      vector[i] <- 3
  } else if ((vector[i] >= 5) && (vector[i] <7)){
      vector[i] <- 2
  } else {
      vector[i] <- 1
  }
}

#replacing the output variable in our dataset with the new variable
redWine <- cbind(redWine, stars = vector)
redWine[,10] <- as.numeric(redWine[,10])

dim(redWine)
sapply(redWine, class)
head(redWine, n=20)
summary(redWine)

# Split out validation dataset
# create a list of 85% of the rows in the original dataset we can use for training
set.seed(7)
validation_index <- createDataPartition(redWine$quality, p=0.85, list=FALSE)
# select 15% of the data for validation
validation <- redWine[-validation_index,]
# use the remaining 85% of data to training and testing the models
datasetTrain <- redWine[validation_index,]


# Summarize data

# dimensions of dataset
dim(datasetTrain)

# list types for each attribute
sapply(datasetTrain, class)


head(datasetTrain, n=20)

# summarize attribute distributions
summary(datasetTrain)


# summarize correlations between input variables
cor(datasetTrain[,1:10])


# Univaraite Visualization

# histograms each attribute
par(mfrow=c(2,7))
for(i in 1:11) {
  hist(redWine[,i], main=names(redWine)[i])
}

# density plot for each attribute
par(mfrow=c(2,7))
for(i in 1:11) {
  plot(density(redWine[,i]), main=names(redWine)[i])
}

# boxplots for each attribute
par(mfrow=c(2,7))
for(i in 1:11) {
  boxplot(redWine[,i], main=names(redWine)[i])
}


# Multivariate Visualizations

# scatterplot matrix
pairs(redWine[,1:10])

# correlation plot
correlations <- cor(redWine[,1:10])
corrplot(correlations, method="circle")


# Evaluate Algorithms

# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
fit.lm <- train(stars~., data=datasetTrain, method="lm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLM
set.seed(7)
fit.glm <- train(stars~., data=datasetTrain, method="glm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLMNET
set.seed(7)
fit.glmnet <- train(stars~., data=datasetTrain, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM
set.seed(7)
fit.svm <- train(stars~., data=datasetTrain, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control)
# CART
set.seed(7)
fit.cart <- train(stars~., data=datasetTrain, method="rpart", metric=metric, preProc=c("center", "scale"), trControl=control)
# kNN
set.seed(7)
fit.knn <- train(stars~., data=datasetTrain, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Compare algorithms
results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(results)
dotplot(results)


# Evaluate Algorithms: with Feature Selection step

# remove correlated attributes
# find attributes that are highly corrected
set.seed(7)
cutoff <- 0.60
correlations <- cor(datasetTrain[,1:10])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)
for (value in highlyCorrelated) {
  print(names(datasetTrain)[value])
}
# create a new dataset without highly corrected features
dataset_features <- datasetTrain[,-highlyCorrelated]
dim(dataset_features)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
fit.lm <- train(stars~., data=dataset_features, method="lm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLM
set.seed(7)
fit.glm <- train(stars~., data=dataset_features, method="glm", metric=metric, preProc=c("center", "scale"), trControl=control)
# GLMNET
set.seed(7)
fit.glmnet <- train(stars~., data=dataset_features, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM
set.seed(7)
fit.svm <- train(stars~., data=dataset_features, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(stars~., data=dataset_features, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale"), trControl=control)
# kNN
set.seed(7)
fit.knn <- train(stars~., data=dataset_features, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Compare algorithms
feature_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(feature_results)
dotplot(feature_results)


# Evaluate Algorithnms: with Box-Cox Transform

# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# lm
set.seed(7)
fit.lm <- train(stars~., data=datasetTrain, method="lm", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# GLM
set.seed(7)
fit.glm <- train(stars~., data=datasetTrain, method="glm", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# GLMNET
set.seed(7)
fit.glmnet <- train(stars~., data=datasetTrain, method="glmnet", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# SVM
set.seed(7)
fit.svm <- train(stars~., data=datasetTrain, method="svmRadial", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(stars~., data=datasetTrain, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale", "BoxCox"), trControl=control)
# kNN
set.seed(7)
fit.knn <- train(stars~., data=datasetTrain, method="knn", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=control)
# Compare algorithms
transform_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(transform_results)
dotplot(transform_results)


# Ensemble Methods
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
# Random Forest
set.seed(7)
fit.rf <- train(stars~., data=datasetTrain, method="rf", metric=metric, preProc=c("BoxCox"), trControl=control)
# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(stars~., data=datasetTrain, method="gbm", metric=metric, preProc=c("BoxCox"), trControl=control, verbose=FALSE)
# Cubist
set.seed(7)
fit.cubist <- train(stars~., data=datasetTrain, method="cubist", metric=metric, preProc=c("BoxCox"), trControl=control)
# Compare algorithms
ensemble_results <- resamples(list(RF=fit.rf, GBM=fit.gbm, CUBIST=fit.cubist))
summary(ensemble_results)
dotplot(ensemble_results)


# look at parameters used for Cubist--best model
print(fit.cubist)


x <- validation[,1:10]
y <- validation[,11]


predictions <- predict(fit.cubist, newdata=x)
print(predictions)

# calculate RMSE
rmse <- RMSE(predictions, y)
r2 <- R2(predictions, y)
print(rmse)

# save the model to disk
saveRDS(fit.cubist, "MyFinalModel2.rds")
#############################################

#use the model for prediction
print("load the model")
model <- readRDS("MyFinalModel2.rds")

# make a predictions on "new data" using the final model
finalPredictions <- predict(model, x)
print(finalPredictions)
rmse <- RMSE(finalPredictions, y)
print(rmse)
#for classification problem only
#confusionMatrix(finalpredictions, y)




