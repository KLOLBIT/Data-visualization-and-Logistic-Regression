# Advertising Dataset - Logistic Regression Analysis

This project uses the **Advertising dataset** (`advertising.csv`) to explore customer behavior and build a **logistic regression model** to predict whether a person will **click on an ad**.

---

##  Project Structure

```
.
├── advertising.csv         # Dataset
├── main.py                 # Analysis & model training script
└── README.md               # Project documentation
```

---

##  Requirements

Install the required Python libraries:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

---

##  Dataset Overview

* **Rows**: 1000
* **Columns**: 10

| Feature                  | Type   | Description                       |
| ------------------------ | ------ | --------------------------------- |
| Daily Time Spent on Site | float  | Time (minutes) user spent on site |
| Age                      | int    | User's age                        |
| Area Income              | float  | Average income of user’s area     |
| Daily Internet Usage     | float  | User’s daily internet usage       |
| Ad Topic Line            | object | Topic of advertisement            |
| City                     | object | User’s city                       |
| Male                     | int    | Gender (0 = Female, 1 = Male)     |
| Country                  | object | User’s country                    |
| Timestamp                | object | Time of data collection           |
| Clicked on Ad            | int    | Target variable (0 = No, 1 = Yes) |

✔️ No missing values were found.

---

##  Exploratory Data Analysis (EDA)

Using **Seaborn**, the following visualizations were created:

1. **Histogram of Age**

   ```python
   sns.histplot(advertising['Age'])
   ```

2. **Jointplot - Age vs Area Income**

   ```python
   sns.jointplot(data=advertising, x='Age', y='Area Income')
   ```

3. **Jointplot KDE - Age vs Daily Time Spent on Site**

   ```python
   sns.jointplot(data=advertising, x='Age', y='Daily Time Spent on Site', kind='kde', color='green')
   ```

4. **Jointplot - Daily Time Spent vs Daily Internet Usage**

   ```python
   sns.jointplot(data=advertising, x='Daily Time Spent on Site', y='Daily Internet Usage')
   ```

5. **Pairplot with Hue = Clicked on Ad**

   ```python
   sns.pairplot(advertising, hue='Clicked on Ad', palette='husl')
   ```

---

##  Model Training

### Features Used:

* `Daily Time Spent on Site`
* `Age`
* `Area Income`
* `Daily Internet Usage`
* `Male`

### Train-Test Split:

```python
from sklearn.model_selection import train_test_split

y = advertising['Clicked on Ad']
X = advertising[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Logistic Regression Model:

```python
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
```

### Predictions:

```python
predictions = lg.predict(X_test)
```

---

##  Model Evaluation

Classification report results:

| Metric    | Class 0 | Class 1 | Accuracy |
| --------- | ------- | ------- | -------- |
| Precision | 0.85    | 0.96    |          |
| Recall    | 0.96    | 0.84    | 0.90     |
| F1-score  | 0.90    | 0.89    |          |

✔️ **Overall accuracy: 90%**

---

##  How to Run

1. Place `advertising.csv` in the working directory.
2. Run the script:

   ```bash
   python main.py
   ```
3. Visualizations and classification metrics will be generated.

---

##  Future Improvements

* Add **feature engineering** (extract insights from `Timestamp`).
* Test other classifiers (RandomForest, SVM, XGBoost).
* Apply **cross-validation** for more robust performance evaluation.

---
