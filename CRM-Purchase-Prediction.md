---

# Predicting Customer Purchase Behavior in an Online Store

---

## Loading Libraries and Uploadind Dataset

```r
# Load required libraries
library(mlbench)
library(rpart)
library(rpart.plot)
library(readr)
library(caret)
library(caTools)
library(randomForest)
```

We load essential libraries for data manipulation, model building (decision tree), and visualization.

```r
# Read and preview data
df <- read_csv("C:/Users/rioss/iCloudDrive/Documents/Hult/Masters/Summer Electives/Forecasting & Predicting the Future Using Data (R)/Class 1/online_shoppers_intention.csv")

View(df)
summary(df)
```
![image](https://github.com/user-attachments/assets/7b41bd46-4858-478d-bc40-009e70d74e5c)


We start by loading the dataset and reviewing its structure to understand the variables and their data types. The summary reveals that some features are on very different scales, which is something we’ll need to address later on.

## Data Cleaning

```r
# Identify successful and unsuccessful customer sessions
which(df$Revenue == TRUE)
business_suc <- df[which(df$Revenue == TRUE),]
business_fail <- df[which(df$Revenue == FALSE),]
```

We separate the customer sessions into two groups: those that led to a purchase (business successes) and those that didn’t (business failures). This step is important because it helps us clearly define which sessions resulted in revenue, in other words, which customers ended up buying something.

```r
# Count how many are TRUE (success) and FALSE (failure)
table(df$Revenue)
```
![image](https://github.com/user-attachments/assets/be095d87-b880-45d0-9449-ae1321d38651)

This code helps me to see how many sessions ended in a purchase (TRUE) versus those that didn’t (FALSE).

```r
# Get percentage breakdown
prop.table(table(df$Revenue)) * 100
```
![image](https://github.com/user-attachments/assets/fadc7f22-086f-406b-b49e-a4150b5766c4)

By calculating the percentages, we can see if the data is balanced or not. In this case, around 85% of the sessions didn’t result in a purchase, while only 15% did, so the dataset is clearly imbalanced.

That’s an important thing to know early on because it affects how we split the data for training and testing later on, and it also tells us that accuracy alone might not be the best metric to evaluate our models.

```r
# Create a binary target variable
# 1 for success, 0 for failure
df$businessoutcome <- ifelse(df$Revenue == TRUE, 1, 0)
```

We create a new column that shows whether a purchase was made or not, using 1 for a business success and 0 for business failure. Many classification algorithms require numerical input for the target variable. This makes it easier for our predictive model to understand the data, so converting a categorical (TRUE/FALSE) outcome into numeric form ensures compatibility and simplifies model training.

```r
# Normalize continuous variables using min-max scaling
rescale <- function(x) {(x - min(x)) / (max(x) - min(x))}
```

The code above defines a function we are calling rescale, which takes a numeric variable and normalizes it using min-max scaling. That means it transforms all the values in the variable so they fall between 0 and 1. We do this because some machine learning models perform better when all the variables are on the same scale, especially if one variable ranges from 0 to 1000 and another from 0 to 1 like in our case (we noticed this when we runned the summary code above). This function helps put everything on a comparable scale.

```r
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

We apply this scale formula to the variables above to normalize them so they are on the same scale. This makes it easier for the model to learn and compare different features without one overpowering the others just because of its size or units.

```r
# We will now convert the months in text to numeric values
df$Month_num <- c() #Creating an empty vector to put results (from 0 to 12)
sort(unique(df$Month)) #Looking at the unique characters in my column Month
```
![image](https://github.com/user-attachments/assets/08dbfba2-b2bb-4d84-9ba2-5b683a4a32de)

This code helps me check how the months are stored in our dataset so I can create a new feature that represents them as numbers. I noticed that there’s no data for January or April in the current dataset, but I’ll still include them in the conversion code just in case future data includes those months.

```r
month_map <- c(Jan=1, Feb=2, Mar=3, Apr=4, May=5, June=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
df$Month_num <- month_map[df$Month]
df$Month_num_norm <- rescale(df$Month_num)
```

We convert the month names (like "Jan", "Feb", etc.) into numbers so our model can understand and work with them, since, as mentioned earlier, many machine learning models can't use text directly. After that, we normalize these numbers to a range between 0 and 1 so that all our data is on a similar scale.

```r
summary(df)
```
![image](https://github.com/user-attachments/assets/220fbb67-80e9-4e92-9830-f6ebb9bd339e)


We run another summary to check that our normalization was properly applied to our data.

### Data Splits

```r
set.seed(123)

# Step 1: Split into 70% training and 30% remaining (val + test)
train_index <- createDataPartition(df$businessoutcome, p = 0.7, list = FALSE)
df_train <- df[train_index, ]
remaining <- df[-train_index, ]

# Step 2: Split remaining 30% into 15% validation and 15% test
set.seed(123)  # optional, to keep it reproducible
val_index <- createDataPartition(remaining$businessoutcome, p = 0.5, list = FALSE)
df_val <- remaining[val_index, ]
df_test <- remaining[-val_index, ]
```
This code splits our dataset into three parts: 70% for training, 15% for validation, and 15% for testing. We use createDataPartition() to perform a stratified split, which ensures the proportion of purchases (1) and non-purchases (0) is preserved in all three subsets. This is especially important since our data is imbalanced, with far fewer successful purchases.

The validation set is key, it allows us to fine-tune model parameters (like the complexity of our decision tree) without touching the test set. This way, we avoid accidentally biasing our model to perform well only on data it has already “seen.” The test set remains completely unseen until the final evaluation step, which gives us a more honest assessment of how well the model will perform in real-world scenarios.

Finally, setting set.seed(123) ensures that the random splits are reproducible, making our workflow consistent and our model comparisons fair.

```r
# Convert target variable to factor in each dataset
df_train$businessoutcome <- as.factor(df_train$businessoutcome)
df_val$businessoutcome <- as.factor(df_val$businessoutcome)
df_test$businessoutcome <- as.factor(df_test$businessoutcome)
```
We use this code because, in classification tasks, it's important to tell R that our target variable (businessoutcome) represents categories, not numbers. By converting it to a factor, we make sure the model treats 1 as a purchase and 0 as no purchase. Many algorithms, like decision trees or random forests, need this format to work properly, otherwise, the model might misinterpret the problem and give incorrect results.
This code splits our dataset into three parts: 70% for training, 15% for validation, and 15% for testing. We use createDataPartition() to perform a stratified split, which ensures the proportion of purchases (1) and non-purchases (0) is preserved in all three subsets. This is especially important since our data is imbalanced, with far fewer successful purchases.


### Target Metric

Before diving into predictions, it’s important to decide which performance metric matters most for our goals.

In this case, I chose to focus on Recall (also called Sensitivity in R). That’s because I believe the main objective here is to catch as many potential business successes as possible, even if that means we might get a few false positives along the way. For example, if we use this model to target customers through email campaigns, we’d rather cast a wider net and possibly reach some people who won’t convert, than risk missing those who would be more likely to make a purchase.

This approach will tie into our final recommendations, but the key takeaway for now is that our models are optimized with Recall as the top priority.

## Predictive Model 1 - Decision Tree

This model is valuable because it doesn’t just make predictions, it also shows you why it's making those predictions through a clear visual structure. That transparency is super helpful for making business decisions.

### Training

```r
# Build the decision tree on the training set
my_tree <- rpart(businessoutcome ~ Administrative + Administrative_Duration +
                 Informational + Informational_Duration + ProductRelated +
                 ProductRelated_Duration + BounceRates + PageValues +
                 SpecialDay + Month,
                 data = df_train,
                 method = "class",
                 cp = 0.01)
```

This code builds a decision tree model to help us predict whether or not a customer will make a purchase (our target variable is businessoutcome, where 1 = purchase, 0 = no purchase). We’re training the model using several input features like how much time a customer spent on different types of pages (Administrative\_Duration, ProductRelated\_Duration, etc.), how often they bounced (BounceRates), how much value the pages had (PageValues), and even what month the session happened in.

The cp=0.01 part sets a complexity parameter that controls how detailed (or deep) the tree can grow—keeping it from becoming overly complex and overfitting the training data.

```r
rpart.plot(my_tree)
```
![image](https://github.com/user-attachments/assets/a10d3ff0-d2d5-4562-a04e-cf130b6f793c)

The code above creates a visualization of the decision tree so we can actually see how the model is making its decisions.

The decision tree starts by checking whether a session's PageValues score is below 3.5. If it is, the model predicts no purchase—this applies to 81% of the data, showing it's a dominant pattern. If the PageValues score is higher, the model explores further, looking at whether it's still under 23. For these mid-range sessions, it considers the month of the visit, suggesting some months (like August, December, or July) may influence purchasing behavior—possibly due to seasonality. It also evaluates how many times users visited administrative pages (like FAQs or policies), where a higher count might signal careful consideration before buying. Finally, BounceRates come into play, with lower bounce rates linked to a higher chance of purchase. Overall, PageValues seems to be the strongest predictor, followed by BounceRates, Month, and Administrative views. This tree offers can help us get actionable insights, like boosting PageValue with stronger CTAs, timing campaigns better, or improving site engagement to lower bounce rates.

However, we need to see how trusworthy this results are, and that is why we will follow the following code, which is going to show us the different metrics based on the performance of our model when working on unseen data. This part is the validation of our model

### Validation

```r

# Predict on the validation set
val_pred <- predict(my_tree, df_val)

# Convert probabilities to class labels (1 if prob > 0.5, else 0)
val_class <- as.factor(as.numeric(val_pred[, 2] > 0.5))

# Evaluate using confusion matrix with "1" as the positive class
confusionMatrix(data = val_class,
                reference = as.factor(df_val$businessoutcome),
                positive = "1")

```
Here’s what the code is doing in simple terms: First, it uses the decision tree to predict the probability that each customer session will lead to a purchase. Then, it translates those probabilities into class labels—if the predicted probability of a purchase is greater than 0.5, it classifies the session as a success (1); otherwise, it's labeled as a non-purchase (0). Finally, it compares these predictions to the actual outcomes from the validation dataset using a confusion matrix, which gives a clear summary of how well the model is performing across different metrics.
![image](https://github.com/user-attachments/assets/62b4c594-71af-44c8-8af8-bce220cef305)


**Analyzing the results**: 

- 97 actual buyers missed (false negatives)
- 176 buyers correctly predicted (true positives)

- Recall / Sensitivity (0.6447):
Out of all the customers who actually made a purchase, the model correctly identified 64% of them. This is our main focus, catching as many real buyers as possible, even if it means making some incorrect guesses.

- Precision (0.6263):
Out of all the customers the model predicted would buy, 63% actually did. This tells us how precise our positive predictions are, how many of them truly lead to a purchase.

- Accuracy (0.8908):
Overall, the model predicts correctly 89% of the time. But since most users don’t buy, accuracy can be misleading—it mostly reflects how well the model identifies non-buyers.

## Decision tree tunned to improve Recall results

Our model is doing a decent job at identifying potential buyers, spotting around 2 out of every 3 actual buyers. But since our main goal is to capture as many potential customers as possible, we’re going to make some adjustments to the decision tree to see if we can boost its performance in terms of Recall.

```r
# Define a range of cp values to test
cp_values <- seq(0.001, 0.05, by = 0.005)

# Store results in a data frame
results <- data.frame(
  cp = numeric(),
  Accuracy = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  Balanced_Accuracy = numeric(),
  Precision = numeric()
)

# Loop through each cp value
for (cp_val in cp_values) {
  model <- rpart(
    businessoutcome ~ Administrative + Administrative_Duration +
      Informational + Informational_Duration + ProductRelated +
      ProductRelated_Duration + BounceRates + PageValues +
      SpecialDay + Month,
    data = df_train,
    method = "class",
    cp = cp_val
  )
  
  # Predict probabilities on validation set
  val_pred <- predict(model, df_val, type = "prob")
  
  # Convert probabilities to class labels (1 if prob > 0.5, else 0)
  val_class <- as.factor(as.numeric(val_pred[, 2] > 0.5))
  
  # Compute confusion matrix
  cm <- confusionMatrix(val_class, as.factor(df_val$businessoutcome), positive = "1")
  
  # Extract all metrics
  acc <- cm$overall["Accuracy"]
  rec <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  bal_acc <- cm$byClass["Balanced Accuracy"]
  prec <- cm$byClass["Precision"]
  
  # Store results
  results <- rbind(results, data.frame(
    cp = cp_val,
    Accuracy = acc,
    Sensitivity = rec,
    Specificity = spec,
    Balanced_Accuracy = bal_acc,
    Precision = prec
  ))
}

# Print full results
print(results)

# Print the cp with the best Sensitivity
best_result <- results[which.max(results$Sensitivity), ]
cat("\nBest cp based on highest Sensitivity:\n")
print(best_result)

```

This code helps us test different values of the complexity parameter (cp) to tune our decision tree and improve its ability to detect actual buyers. Since our main focus is Recall, or in R, Sensitivity, we want to find the cp value that helps the model catch the most true positives (purchasing customers), even if that means sacrificing a bit of precision or accuracy.

The model tries out 10 different cp values, evaluates each one on the validation set, and tracks key metrics like Accuracy, Sensitivity (Recall), Specificity, Balanced Accuracy, and Precision. After looping through all values, it picks the model that had the highest Sensitivity.
![image](https://github.com/user-attachments/assets/f1116d00-5f8f-4336-80ef-cf0d84669685)

**Analyzing the results**: 

The best model came from cp = 0.046, which gave us a Sensitivity of 0.7326, meaning it correctly identifies about 73% of buyers.

This is a noticeable improvement over our previous recall (~64%) before tuning the cp parameter.

While overall accuracy dropped slightly (from ~89% to ~86%), that’s expected and acceptable because our goal is to reduce missed sales, not necessarily be perfect overall.

Precision (how many of the predicted buyers actually buy) went down a bit to ~53%, which reflects the trade-off we’re making: we’re okay with some false positives if it means catching more actual buyers.

### Testing

```r

# Predict on the validation set
test_pred <- predict(my_tree, df_test)

# Convert probabilities to class labels (1 if prob > 0.5, else 0)
test_class <- as.factor(as.numeric(test_pred[, 2] > 0.5))

# Evaluate using confusion matrix with "1" as the positive class
confusionMatrix(data = test_class,
                reference = as.factor(df_test$businessoutcome),
                positive = "1")
```
![image](https://github.com/user-attachments/assets/c0a8e507-eaa0-43ac-943a-de68da713574)


When we tested the model on completely new data, the recall dropped a bit, which is something we often expect. On the validation set, the model was identifying about 73 out of every 100 buyers. On the test set, that dropped to around 62, meaning it's still doing a solid job, just not quite as strong as before.

This slight drop is totally normal, it simply means the model doesn't perform exactly the same on new data, but overall it's still holding up well.

Logically, when recall goes down, precision tends to go up, and that’s exactly what we’re seeing here, the model is more accurate when it predicts a buyer, even if it's catching a few less overall.

While it’s always ideal to keep recall high, being able to correctly identify 62% of potential customers is still a big win for any business. It gives valuable insight into buyer behavior and opens the door to smarter marketing and engagement strategies.

## Predictive Model 2 - Random Forest

We will now create a Random Forest model, which builds many decision trees using different random subsets of the data (both rows and columns). By combining the results of these trees, we are aiming to get more accurate and stable predictions than the single tree created above.

### Training

``` r
# Train a Random Forest model
my_forest <- randomForest(x=df_train[, c('Administrative', 'Administrative_Duration',
                                         'Informational', 'Informational_Duration', 'ProductRelated',
                                         'ProductRelated_Duration', 'BounceRates', 'PageValues',
                                         'SpecialDay', 'Month')],
                          y=df_train$businessoutcome,
                          ntree=500)
```                          
This code trains a Random Forest model to predict whether a customer will make a purchase (businessoutcome). It does this by using several input features like time spent on different pages, bounce rates, and page values. The model creates 500 individual decision trees, each built using slightly different samples of the data and different combinations of variables. By combining the predictions from all these trees, the Random Forest can make more accurate and stable predictions than a single decision tree. This method helps reduce errors and overfitting, making it a strong choice for classification problems like this one.

It is important to note that I chose to use 500 trees (ntree = 500) because my dataset is relatively small, and I wanted to get more stable and reliable results. However, if we were working with a much larger dataset, training that many trees could become computationally expensive. In most real world scenarios, using around 100 trees is usually enough to get strong performance without overloading our system.

### Validating

```r
# Predict probabilities on the validation set
forest_val_pred <- predict(my_forest, df_val, type = "prob")

# Convert probabilities to class labels (1 if prob > 0.5, else 0)
forest_val_class <- as.factor(as.numeric(forest_val_pred[, 2] > 0.5))

# Evaluate using confusion matrix with "1" as the positive class
confusionMatrix(data = forest_val_class,
                reference = as.factor(df_val$businessoutcome),
                positive = "1")
```
![image](https://github.com/user-attachments/assets/36ca6918-d65c-4ac0-abec-dd15f6b4152d)

**Analyzing the results**: 

- 45 actual buyers missed (false negatives)

- 223 buyers correctly predicted (true positives)

- Recall / Sensitivity (0.8321):
The model correctly identified 83% of all the customers who actually made a purchase. This is a strong result and shows the model is doing a great job at capturing real buyers—our main goal in this project.

- Precision (0.9253):
Out of all the customers the model predicted would buy, about 93% actually did. That means the model is not only good at catching buyers, but it’s also very accurate in its predictions, with very few false alarms.

- Accuracy (0.9659):
Overall, the model got nearly 97% of its predictions right. While accuracy can sometimes be misleading in imbalanced datasets, in this case, it’s consistent with the high recall and precision, indicating strong performance across the board.

- Balanced Accuracy (0.9104):
This metric averages how well the model performs on both buyers and non-buyers. At 91%, it confirms the model is doing well even in the presence of class imbalance.

```r
# Plot feature importance
plot(my_forest)
varImpPlot(my_forest)
```
![image](https://github.com/user-attachments/assets/9e5618de-7031-4c11-b613-d688c5a09039)

This plot shows which features were most important in our Random Forest model for predicting purchases. The higher the bar, the more helpful the variable was in making decisions. We can take away the following insights from it:

- PageValues was by far the most influential, customers who engage with high-value pages are more likely to buy.

- ProductRelated_Duration and ProductRelated also played key roles, showing that time spent and number of product page visits matter.

- BounceRates and Administrative_Duration were moderately important, these reflect user engagement and careful consideration.

- Features like SpecialDay and Informational pages had less impact on predictions.

## Predictive Model 3 - Neural Networks

...IN PROGRESS

## Conclusion and Recommendations

...IN PROGRESS
