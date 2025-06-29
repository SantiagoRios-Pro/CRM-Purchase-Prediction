CRM-Purchase-Prediction
================
Santiago Rios Castro
2025-06-28

------------------------------------------------------------------------

``` r
# Load required libraries
library(mlbench)
library(rpart)
library(rpart.plot)
library(readr)
```

We load essential libraries for data manipulation, model building
(decision tree), and visualization.

``` r
# Read and preview data
df <- read_csv("path/to/online_shoppers_intention.csv")
View(df)
summary(df)
```

We read in the dataset and examine its structure to understand the
variables and data types.

``` r
# Identify successful and unsuccessful customer sessions
business_suc <- df[which(df$Revenue == TRUE),]
business_fail <- df[which(df$Revenue == FALSE),]
```

We separate customer sessions that resulted in revenue from those that
did not.

``` r
# Create a binary target variable
# 1 for success, 0 for failure
df$businessoutcome <- ifelse(df$Revenue == TRUE, 1, 0)
```

We create a new binary column to simplify classification: success (1) or
not (0).

``` r
# Normalize continuous variables using min-max scaling
rescale <- function(x) {(x - min(x)) / (max(x) - min(x))}

# Apply rescaling to selected variables
df$Administrative_norm <- rescale(df$Administrative)
df$Administrative_Duration_norm <- rescale(df$Administrative_Duration)
df$Informational_norm <- rescale(df$Informational)
df$Informational_Duration_norm <- rescale(df$Informational_Duration)
df$ProductRelated_norm <- rescale(df$ProductRelated)
df$ProductRelated_Duration_norm <- rescale(df$ProductRelated_Duration)
df$BounceRates_norm <- rescale(df$BounceRates)
df$PageValues_norm <- rescale(df$PageValues)
df$SpecialDay_norm <- rescale(df$SpecialDay)
```

We normalize continuous variables so they are on the same scale, which
improves model performance.

``` r
# Convert month names to numeric values
month_map <- c(Jan=1, Feb=2, Mar=3, Apr=4, May=5, June=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
df$Month_num <- month_map[df$Month]
df$Month_num_norm <- rescale(df$Month_num)
```

We convert month names to numeric values and normalize them.

``` r
# Split data into training (80%) and test (20%) sets
set.seed(123)
indx <- sample(x=1:nrow(df), size=0.8*nrow(df))
df_train <- df[indx,]
df_test <- df[-indx,]
```

We split the dataset for training and testing to evaluate model
performance.

``` r
# Build a decision tree using Gini index
my_tree <- rpart(businessoutcome ~ Administrative + Administrative_Duration +
                 Informational + Informational_Duration + ProductRelated +
                 ProductRelated_Duration + BounceRates + PageValues +
                 SpecialDay + Month,
                 data=df_train, method="class", cp=0.01)

rpart.plot(my_tree)
```

We train a decision tree to predict purchase outcomes and visualize the
model.

``` r
# Predict and evaluate decision tree
library(caret)
tree_pred <- predict(my_tree, df_test)
confusionMatrix(data=as.factor(as.numeric(tree_pred[,2] > 0.5)),
                reference=as.factor(df_test$businessoutcome))
```

We evaluate the model’s accuracy using a confusion matrix.

``` r
# Install and load Random Forest libraries
install.packages("caTools")
install.packages("randomForest")
library(caTools)
library(randomForest)
```

We install and load additional libraries to build a Random Forest model.

``` r
# Train a Random Forest model
my_forest <- randomForest(x=df_train[, c('Administrative', 'Administrative_Duration',
                                         'Informational', 'Informational_Duration', 'ProductRelated',
                                         'ProductRelated_Duration', 'BounceRates', 'PageValues',
                                         'SpecialDay', 'Month')],
                          y=df_train$businessoutcome,
                          ntree=500)

# Predict and evaluate Random Forest
forest_pred <- predict(my_forest, df_test)
confusionMatrix(data=as.factor(as.numeric(forest_pred > 0.5)),
                reference=as.factor(df_test$businessoutcome))

# Plot feature importance
plot(my_forest)
varImpPlot(my_forest)
```

We build a more robust model using Random Forest, evaluate its
performance, and analyze which features are most important for
predicting purchases.
