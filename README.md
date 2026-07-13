# Student Test Score Prediction

## Project Overview

This project analyzes student performance data and builds a regression model to predict posttest scores using information available before the final test.

The main objective is to support teachers and schools in identifying students who may require additional academic support.

The model is intended as a decision-support tool. It should not automatically assign students to groups or make educational decisions.

---

## Educational Problem

Teachers often need to identify students who may struggle before a final assessment takes place.

The main project question is:

> Can a student's posttest score be predicted using their pretest result and basic information about the school and teaching environment?

The model can support teachers by:

* predicting expected final scores,
* identifying students with the lowest predicted results,
* prioritizing students who may need additional support,
* supporting earlier educational interventions.

---

## Dataset

The dataset contains information about **2,133 students** and includes **11 variables**.

Each row represents one student.

### Dataset Variables

* `school` — identifier of the school attended by the student,
* `school_setting` — location of the school, such as urban, suburban, or rural,
* `school_type` — type of school, such as public or non-public,
* `classroom` — identifier of the classroom or class group,
* `teaching_method` — teaching method used in the classroom,
* `n_student` — number of students in the classroom,
* `student_id` — unique student identifier,
* `gender` — student gender,
* `lunch` — type of school lunch program,
* `pretest` — student score before the teaching period,
* `posttest` — student score after the teaching period and the prediction target.

The dataset contains:

* 2,133 observations,
* no missing values,
* no duplicated rows,
* 2,133 unique student identifiers.

---

## Project Workflow

### 1. Preliminary Data Analysis

The dataset is examined to check:

* dimensions,
* data types,
* descriptive statistics,
* missing values,
* duplicate records,
* unique student identifiers.

The target variable, `posttest`, ranges from 32 to 100 points.

Its main descriptive statistics are:

* mean: approximately 67.1,
* median: 68,
* standard deviation: approximately 14.0,
* interquartile range: 56–77.

The model predicts a numerical score rather than only assigning students to low- and high-score categories.

---

### 2. Exploratory Data Analysis

The exploratory analysis examines relationships between student performance and selected variables.

#### Pretest and posttest

Pretest and posttest scores have a very strong positive correlation:

```text
Correlation: approximately 0.95
```

This indicates that the pretest score is likely to be the strongest predictor of the final result.

Students with higher initial scores generally also achieve higher posttest scores.

#### Teaching method

Average posttest results differ between teaching methods:

| Teaching method | Mean posttest score |
| --------------- | ------------------: |
| Experimental    |               72.98 |
| Standard        |               63.85 |

Students taught using the experimental method achieve higher average scores.

However, this does not prove that the teaching method itself causes the difference. The groups may also differ in initial knowledge, school characteristics, or other variables.

#### Lunch status

Average posttest scores also differ by lunch-program status:

| Lunch status                     | Mean posttest score |
| -------------------------------- | ------------------: |
| Does not qualify                 |               74.38 |
| Qualifies for reduced/free lunch |               57.48 |

Lunch status may act as an indirect indicator of socioeconomic conditions. It should not be interpreted as a direct cause of student performance.

#### School setting

Average posttest scores vary across school settings:

| School setting | Mean posttest score |
| -------------- | ------------------: |
| Suburban       |               76.04 |
| Rural          |               64.05 |
| Urban          |               61.75 |

Students from suburban schools achieve the highest average scores.

These differences may also be related to other factors included in the dataset.

#### Class size

Class size has a moderate negative correlation with posttest score:

```text
Correlation: approximately -0.50
```

Students in smaller classes tend to achieve higher scores.

This relationship does not establish causality because class size may also be related to school type, school setting, or other characteristics.

---

## Data Preparation

The target variable is:

```python
posttest
```

### Features used for prediction

#### Numerical features

* `n_student`,
* `pretest`.

#### Categorical features

* `school_setting`,
* `school_type`,
* `teaching_method`,
* `gender`,
* `lunch`.

### Variables excluded from the model

* `student_id` is excluded because it is only an identifier.
* `school` and `classroom` are not used as ordinary predictive features because the model could memorize specific schools or classrooms instead of learning general relationships.

---

## Train-Test Split

The dataset is split using `GroupShuffleSplit`.

The grouping variable is `school`.

This ensures that schools included in the test set do not appear in the training set.

The number of schools shared between the training and test sets is:

```text
0
```

This creates a more demanding evaluation scenario because the model is tested on students from schools it has not seen during training.

---

## Preprocessing

Preprocessing and model training are combined in scikit-learn pipelines.

### Numerical preprocessing

Numerical variables are processed using:

* `SimpleImputer`,
* `StandardScaler`.

Although the current dataset has no missing values, imputation is included as protection against missing values in future data.

### Categorical preprocessing

Categorical variables are processed using:

* `SimpleImputer`,
* `OneHotEncoder`.

This allows categorical variables to be used by regression models.

---

## Models

The following regression models are trained and compared:

1. `DummyRegressor`,
2. `LinearRegression`,
3. `Ridge`,
4. `RandomForestRegressor`,
5. `GradientBoostingRegressor`.

The Dummy Regressor serves as a baseline.

Linear Regression and Ridge represent simple linear approaches, while Random Forest and Gradient Boosting can model more complex non-linear relationships.

---

## Evaluation Metrics

The models are evaluated using:

### Mean Absolute Error

MAE measures the average absolute difference between the actual and predicted score.

It is easy to interpret because it uses the same unit as the target variable: test points.

### Root Mean Squared Error

RMSE gives additional weight to larger prediction errors.

### R²

R² measures how much of the variation in posttest scores is explained by the model.

A higher R² indicates better predictive performance.

---

## Model Results

| Model             |      MAE |     RMSE |        R² |
| ----------------- | -------: | -------: | --------: |
| Linear Regression | **2.59** | **3.26** | **0.958** |
| Ridge             |     2.59 |     3.26 |     0.958 |
| Random Forest     |     3.49 |     4.71 |     0.913 |
| Gradient Boosting |     3.58 |     4.74 |     0.912 |
| Dummy Regressor   |    13.78 |    16.09 |    -0.016 |

Linear Regression achieves the best result.

Its performance on students from schools not included in the training data is:

* MAE: approximately 2.59 points,
* RMSE: approximately 3.26 points,
* R²: approximately 0.96.

This means that the predicted score differs from the actual score by approximately 2.6 points on average.

More complex models do not improve performance. The strong, nearly linear relationship between pretest and posttest makes Linear Regression the most appropriate model in this project.

---

## Selected Model Evaluation

Linear Regression is selected as the final model.

The predicted and actual posttest scores are compared using a scatter plot.

Most observations are located close to the ideal prediction line, indicating that the predicted values are generally close to the actual scores.

The distribution of residuals is concentrated around zero.

This suggests that the model does not show a strong general tendency to:

* systematically overestimate scores,
* systematically underestimate scores.

---

## Students Requiring Support

The regression model is also used to identify students who may need additional support.

For demonstration purposes, a student is treated as being at risk when:

```python
posttest < 60
```

The 60-point threshold is only an example. In a real educational setting, it should be defined together with teachers and adapted to the test scale and educational requirements.

### Risk-classification results

|                      | Predicted not at risk | Predicted at risk |
| -------------------- | --------------------: | ----------------: |
| Actually not at risk |                   308 |                 5 |
| Actually at risk     |                     3 |               148 |

The model correctly identifies 148 of 151 students whose actual score is below 60.

---

## Support Capacity Analysis

Students are ranked from the lowest to the highest predicted posttest score.

The project examines the 20% of students with the lowest predicted scores.

Results:

* `Precision@20%`: 1.00,
* `Recall@20%`: approximately 0.616.

This means that:

* every student selected within the bottom 20% actually belongs to the at-risk group,
* the selected group includes approximately 62% of all students whose actual posttest result is below 60.

This analysis is useful when the school has limited support capacity and cannot provide additional help to every student.

---

## Feature Importance

Permutation importance is used to evaluate how strongly each variable contributes to model performance.

The most important feature is:

```text
pretest
```

The next most useful variable is:

```text
teaching_method
```

The remaining features contribute much less to prediction quality.

Approximate importance ranking:

1. `pretest`,
2. `teaching_method`,
3. `n_student`,
4. `lunch`,
5. `gender`,
6. `school_type`,
7. `school_setting`.

This suggests that a relatively simple system based mainly on the pretest score may perform almost as well as a more complex model.

Feature importance describes predictive usefulness and does not prove causality.

---

## Error Analysis

The students with the largest absolute prediction errors are examined separately.

Large errors may occur when a student performs much better or much worse than their pretest result suggests.

Possible explanations include variables not available in the dataset, such as:

* motivation,
* attendance,
* additional tutoring,
* absence,
* learning progress,
* completed assignments,
* individual educational support.

These cases demonstrate why the model should complement teacher judgement rather than replace it.

---

## Recommendations

1. Use Linear Regression as the primary model because it provides the best result and is easy to explain.

2. Treat the pretest score as the most important predictor of the final result.

3. Use predicted scores to create a ranked list of students who may require additional support.

4. Present predictions together with the typical model error of approximately 2.6 points.

5. Do not use the model to make automatic educational decisions or assign students to groups without teacher review.

6. Extend future datasets with information such as attendance, platform activity, completed assignments, and score changes over time.

---

## Key Conclusions

* Pretest and posttest scores are very strongly correlated.
* Linear Regression achieves the best predictive performance.
* The model explains approximately 96% of the variation in posttest scores.
* Predictions differ from actual scores by approximately 2.6 points on average.
* The pretest score is by far the most important predictive variable.
* More complex models do not outperform the simpler linear model.
* The model can help teachers prioritize students who may need support.
* Predictions should always be interpreted together with teacher knowledge and professional judgement.

---

## Technologies

* Python
* pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn
* Jupyter Notebook

## Limitations

* The dataset contains a limited number of variables describing student performance.
* Important factors such as attendance, motivation, completed assignments, and additional support are not included.
* The relationships observed in the dataset do not establish causality.
* The 60-point support threshold is only a demonstration example.
* The model is evaluated on one group-based train-test split.
* Results may differ for schools, students, or educational systems not represented in the dataset.
* A very strong relationship between pretest and posttest may reduce the additional value of other features.

---

## Possible Future Improvements

* add attendance and learning-activity variables,
* include completed assignments and homework results,
* analyze student progress over time,
* use repeated group cross-validation,
* add prediction intervals,
* evaluate model fairness across student groups,
* create a dashboard for teachers,
* deploy the model as an API,
* test whether model-supported interventions improve student outcomes.

---

## Author

Artur Strąg
