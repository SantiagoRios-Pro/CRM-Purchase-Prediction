library(mlbench)
library(rpart)
library(rpart.plot)
library(readr)

df <- read_csv("C:/Users/rioss/iCloudDrive/Documents/Hult/Masters/Summer Electives/Forecasting & Predicting the Future Using Data (R)/Class 1/online_shoppers_intention.csv")
View(df)

summary(df)

# Identifying all the business successes

which(df$Revenue == TRUE)
business_suc <- df[which(df$Revenue == TRUE),]
business_fail <- df[which(df$Revenue == FALSE),]

# Example 2: convert the TRUE Rev customers into 1
# and convert the FALSE Rev customers into 0

df$businessoutcome <- c() #Creating an empty vector to put results (0 and 1)

df[which(df$Revenue == TRUE), c('businessoutcome')] <- 1
df[which(df$Revenue == FALSE), c('businessoutcome')] <- 0

# Massaging our variables to make them more similar
# using a min-max rescaling / normalization

rescale <- function(x){ # x is a dataframe variable
  min_max <- (x-min(x))/(max(x)-min(x))
  return(min_max)
} # closing the rescale function

  # We will call our function to massage the variable (normalize them) 
  df$Administrative_norm <- rescale(x=df$Administrative)
  df$Administrative_Duration_norm <-rescale(x=df$Administrative_Duration)
  df$Informational_norm <-rescale(x=df$Informational)
  df$Informational_Duration_norm <-rescale(x=df$Informational_Duration)
  df$ProductRelated_norm <-rescale(x=df$ProductRelated)
  df$ProductRelated_Duration_norm <-rescale(x=df$ProductRelated_Duration)
  df$BounceRates_norm <-rescale(x=df$BounceRates)
  df$PageValues_norm <-rescale(x=df$PageValues)
  df$SpecialDay_norm <-rescale(x=df$SpecialDay)

# We will now convert the months in text to numeric values
df$Month_num <- c() #Creating an empty vector to put results (from 0 to 12)

sort(unique(df$Month)) #Looking at the unique characters in my column Month
table(df$Month) #Looking at my characters from Month and how many rows there are with them
# We can do pretty much both codes to get to the same output,


  df[which(df$Month == 'Jan'), c('Month_num')] <- 1
  df[which(df$Month == 'Feb'), c('Month_num')] <- 2
  df[which(df$Month == 'Mar'), c('Month_num')] <- 3
  df[which(df$Month == 'Apr'), c('Month_num')] <- 4
  df[which(df$Month == 'May'), c('Month_num')] <- 5
  df[which(df$Month == 'June'), c('Month_num')] <- 6
  df[which(df$Month == 'Jul'), c('Month_num')] <- 7
  df[which(df$Month == 'Aug'), c('Month_num')] <- 8
  df[which(df$Month == 'Sep'), c('Month_num')] <- 9
  df[which(df$Month == 'Oct'), c('Month_num')] <- 10
  df[which(df$Month == 'Nov'), c('Month_num')] <- 11
  df[which(df$Month == 'Dec'), c('Month_num')] <- 12

df$Month_num_norm <-rescale(x=df$Month_num)

summary(df)

# Coding random sample
# Step 1:
indx <- sample(x=1:nrow(df), size=0.8*nrow(df))
# Step 2: using indx to subset our data
df_train <- df[indx,]
df_test <- df[-indx,]

# Building our first predicting model - Classification Model
library(rpart)
library(rpart.plot)
# gini-tree
my_tree <- rpart(businessoutcome ~ Administrative + Administrative_Duration +
        Informational + Informational_Duration +ProductRelated + 
        ProductRelated_Duration + BounceRates + PageValues + SpecialDay + Month,
      data=df_train, method="class", cp=0.01)

rpart.plot(my_tree)

# Using the tree to predict on testing data
tree_pred <- predict(my_tree, df_test)

# Checking how good my predictions are
# running performance testing
library(caret)
confusionMatrix(data=as.factor(as.numeric(tree_pred[,2]>0.5)) ,
                reference = as.factor(as.numeric(df_test$businessoutcome)))

# Build a "challenger" model
# Random Forest

install.packages("caTools")
install.packages("randomForest")

library(caTools)
library(randomForest)

my_forest <- randomForest(x=df_train[, c('Administrative', 'Administrative_Duration',
                          'Informational', 'Informational_Duration', 'ProductRelated' 
                          ,'ProductRelated_Duration', 'BounceRates', 'PageValues' 
                          ,'SpecialDay', 'Month')], 
             y=df_train$businessoutcome, 
             ntree=500)

my_forest
forest_pred <- predict(my_forest, df_test)

# Confusion Matrix:

confusionMatrix(data=as.factor(as.numeric(forest_pred>0.5)) ,
                reference = as.factor(as.numeric(df_test$businessoutcome)))

plot(my_forest)

# Looking at variable importance:
varImpPlot(my_forest)
