# NHL-Games-Winning-Losing-Prediction

For this project, I used Kaggle’s NHL Game dataset to build various classification models to predict whether a particular team will win or not. For the purpose of this project, I converted the output to a binary output where each team either “win” or not . The output of the game is determined by 10 input variables:
```
Shots
Hits
Pim
PowerPlayOpportunities
PowerPlayGoals
FaceOffWinPercentage
Giveaways
Takeaways
Blocked
Goals
```
Objectives

The objectives of this project are as follows:

```
To experiment with different regression methods to see which yields the highest accuracy
To determine which features are the most indicative of a game output
```
Project Outline:
```
- Data Preparation
- Data Wrangling
- Data Exploration
- Model Development
- Model Evaluation and Refinement
```

Pre Processing Steps Followed:
```
1. Identifying Null Values.
2. Plotting Histogram to find the Data Point's Distribution.
3. For each feature, identifying Outliers.
4. Outliers Treatment using IQR Method for each feauture.
```

Steps included in this project:

```
Importing Lib
Loading Data
Understanding Data
Missing Values
Exploring Variables(Data Anylasis)
Feature Selection
Correlation Matrix is formed.
Dropping columns that doesn't affect the output variable.
One-Hot Encoding categorical variables in columns that affect the output variable.
Train, Test Split is done.
Feature Scaling is performed.
Applying different models
Choosing right model
```





