#--------------------BA WITH R PROJECT---------------


rm(list=ls())
cat("\014")
install.packages("ggplot2")
library(dplyr)
library(ggplot2)
setwd("/Users/hp/Downloads")
heart <- read.csv("heart.csv")
# Set custom color palette
my_colors <- c("#E69F00", "#56B4E9")
heart <- heart %>%
  mutate(cp = as.factor(cp),
         restecg = as.factor(restecg),
         slope = as.factor(slope),
         ca = as.factor(ca),
         thal = as.factor(thal),
         sex = factor(sex, levels = c(0,1), labels = c("female", "male")),
         fbs = factor(fbs, levels = c(0,1), labels = c("False", "True")),
         exang = factor(exang, levels = c(0,1), labels = c("No", "Yes")),
         target = factor(target, levels = c(0,1), labels = c("no heart disease","Heart disease")))

#1)How does age and gender affect the likelihood of having heart disease?
# Create a stacked bar chart
ggplot(heart, aes(x = age, fill = factor(sex))) +
  geom_bar() +
  labs(title = "Relationship between Age, Gender, and Heart Disease",
       x = "Age",
       y = "Count",
       fill = "Gender") +
  facet_grid(cols = vars(target))

#2)Is there a correlation between resting blood pressure and the presence of heart disease?
ggplot(heart, aes(x = target, y = trestbps)) +
  geom_boxplot() +
  labs(title = "Resting Blood Pressure Distribution by Heart Disease",
       x = "Presence of Heart Disease",
       y = "Resting Blood Pressure") +
  scale_x_discrete(labels = c("Absent", "Present")) +
  theme_minimal()

#3)Are certain types of chest pain (cp) more strongly associated with heart disease than others?
library(ggplot2)
# create a data frame with the counts of each cp type for each target value
cp_counts <- heart %>%
  group_by(cp, target) %>%
  summarize(count = n()) %>%
  ungroup()

# create a stacked bar plot
ggplot(cp_counts, aes(x = cp, y = count, fill = factor(target))) +
  geom_col(position = "stack") +
  labs(title = "Association between Chest Pain Types and Heart Disease",
       x = "Chest Pain Type",
       y = "Count") +
  scale_fill_manual(values = c("#F8766D", "#00BFC4"),
                    labels = c("Absent", "Present")) +
  theme_minimal()

#4)How does the presence of heart disease relate to the maximum heart rate achieved during exercise (thalach)?
library(ggplot2)
ggplot(heart, aes(x=thalach , fill=target)) + geom_histogram(aes(y=..density..), color="black") +
  geom_density(alpha=.2, fill="green")+
  facet_wrap(~target, ncol=1,scale="fixed")+
  xlab("Maximum Heart Rate Achieved") + 
  ylab("Density/Count") +
  ggtitle("Maximum Heart Rate Achieved") +
  scale_fill_discrete(name = "Heart Disease", labels = c("Absence", "Presence")) + theme(plot.title = element_text(hjust = 0.5))

#5)Are there any values of the thalassemia blood disorder (thal) that are more strongly associated with heart disease?
ggplot(heart, aes(x = thal, fill = factor(target))) +
  geom_bar(position = "dodge") +
  labs(title = "Association between Thalassemia and Heart Disease",
       x = "Thalassemia Type",
       y = "Count") +
  scale_fill_discrete(name = "Presence of Heart Disease", labels = c("Absent", "Present")) +
  theme_minimal()

#6)Are there any correlations between the number of major vessels (ca) colored by flouroscopy and the presence of heart disease?
library(ggplot2)
# Create a grouped bar chart of heart disease vs. number of major vessels
ggplot(heart, aes(x = ca, fill = factor(target))) +
  geom_bar(position = "dodge") +
  labs(title = "Correlation between Number of Major Vessels and Heart Disease",
       x = "Number of Major Vessels",
       y = "Count") +
  scale_fill_manual(values = c("gray80", "dodgerblue3"), labels = c("Absent", "Present")) +
  theme_minimal()


#7)How do different features of the dataset relate to one another, and what are the strongest predictors of heart disease?
heart_no_factors <- read.csv("heart.csv")
install.packages("ggcorrplot")
library(ggcorrplot)
library(ggcorrplot)
library(RColorBrewer)
heart_no_factors[] <- lapply(heart_no_factors, as.numeric)

#a)correlation matrix and heatmap
heart_corr <- cor(heart_no_factors[,1:13])
ggcorrplot(heart_corr, type = "upper", hc.order = TRUE, 
           colors = c("#6D9EC1", "white", "#E46726"), 
           title = "Correlation Matrix of Heart Dataset")



#b)Recursive feature elimination with cross-validation (RFECV):
install.packages("randomForest")
library(randomForest)
library(caret)
ctrl <- rfeControl(functions=rfFuncs, method="cv", number=10)
lmProfile <- rfe(heart[,1:13], heart$target, sizes=c(1:13), rfeControl=ctrl)
lmProfile

#c)Principal Component Analysis (PCA):
library(factoextra)
library(dplyr)
heart_std <- scale(heart_no_factors[,1:13])
pca <- princomp(heart_std, cor = TRUE)
PC <- as.data.frame(pca$scores) %>%
  mutate_all(~ round(., 2))
prop_var <- round(pca$sdev^2 / sum(pca$sdev^2), 2)
fviz_eig(pca, addlabels = TRUE)
fviz_pca_biplot(pca, label = "var", col.var = "black", repel = TRUE)


#d)Random Forest:
library(randomForest)
set.seed(123)
heart_rf <- randomForest(heart_no_factors$target ~ ., data = heart_no_factors[,1:13], ntree=500, importance=TRUE)
varImpPlot(heart_rf)


#8)Can we build a predictive model to accurately predict the presence of heart disease based on the available features?
# Training and testing the models
# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(heart_no_factors$target, p = 0.8, list = FALSE)
trainData <- heart_no_factors[trainIndex, ]
testData <- heart_no_factors[-trainIndex, ]


#----------------dt-----------
heart <- read.csv("heart.csv")
library(caret)
library(dplyr)
heart <- na.omit(heart)

# Convert non-numeric variables to factors
heart$sex <- as.factor(heart$sex)
heart$cp <- as.factor(heart$cp)
heart$fbs <- as.factor(heart$fbs)
heart$restecg <- as.factor(heart$restecg)
heart$exang <- as.factor(heart$exang)
heart$target <- as.factor(heart$target)

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(heart$target, p = .8, list = FALSE, times = 1)
trainingData <- heart[trainIndex, ]
testingData <- heart[-trainIndex, ]

# Train decision tree model
install.packages("rpart.plot")
library(rpart)
tree_model <- rpart(target ~ ., data = trainingData, method = "class")

# Make predictions on testing set
tree_pred <- predict(tree_model, testingData, type = "class")
tree_RMSE <- RMSE(as.numeric(as.character(tree_pred)), as.numeric(as.character(testingData$target)))
tree_R2 <- R2(as.numeric(as.character(tree_pred)), as.numeric(as.character(testingData$target)))
cat(paste("Decision Tree RMSE = ", tree_RMSE))
cat(paste("Decision Tree R-squared = ", tree_R2))
##
target.pred <- predict(tree_model, trainingData, type="class")
# extract the actual class of each observation in trainingData
target.actual <- trainingData$target

## Confusion Matrix for Decision Tree
confusion.matrix <- table(target.pred, target.actual)
confusion.matrix
addmargins(confusion.matrix)
pt <- prop.table(confusion.matrix)  
pt
# accuracy
accuracy<-pt[1,1] + pt[2,2]
accuracy
# Boosting
install.packages("adabag")
library(adabag)
library(rpart) 
library(caret)
fit.boosting <- boosting(target ~ ., data = trainingData, mfinal = 20)
pred <- predict(fit.boosting, testingData, type = "class")
cm1 <- confusionMatrix(as.factor(pred$class), testingData$target)
cm1
# Bagging
fit.bagging <- bagging(target ~ ., data = trainingData, mfinal = 20)
pred <- predict(fit.bagging, testingData, type = "class")
cm2 <- confusionMatrix(as.factor(pred$class), testingData$target)
cm2 

### 
# Logistic regression
logit.reg <- glm(target ~ ., 
                 data = trainingData, family = "binomial") 
summary(logit.reg)

logitPredict <- predict(logit.reg, testingData, type = "response")
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)
actual <- testingData$target
predict <- logitPredictClass
cm <- table(predict, actual)

# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy for Logistic Regression
a=(tp + tn)/(tp + tn + fp + fn)
a

## Naive Bayes
install.packages("e1071")
library(e1071)

# run naive bayes
fit.nb <- naiveBayes(target ~ ., 
                     data = trainingData)

# Evaluate Performance using Confusion Matrix
actual <- testingData$target
# predict class probability
nbPredict <- predict(fit.nb, testingData, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, testingData, type = "class")
cm <- table(nbPredictClass, actual)
cm 
# alternative way to get confusion matrix
cm3<-confusionMatrix(nbPredictClass, actual, positive="1")
cm3

## Comparing Accuracy of all models
result <- rbind(cm1$overall["Accuracy"], cm2$overall["Accuracy"],a,accuracy,cm3$overall["Accuracy"])
row.names(result) <- c("boosting", "bagging","logistic regression","Decision Tree","Naive Bayes")
result

