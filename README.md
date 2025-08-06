 

# **Data Preprocessing using Python and scikit-learn**

This program demonstrates major steps in the process of prepping a dataset to use with machine learning using Python libraries such as NumPy, Pandas, and scikit-learn. It addresses missing data, encoding categorical features, and dividing the dataset into training and test sets.

---

## **Libraries Used**

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

* **NumPy**: For numerical computations.
* **Matplotlib**: Added for potential visualizations (not used in this program).
* **Pandas**: To read and work with data in table format.

---

## **1. Importing the Dataset**

```python
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

* Imports the dataset from a CSV file named `Data.csv`.
* `x` contains all the independent variables (input variables).
* `y` contains the dependent variable (target variable).

---

## **2. Handling Missing Data**

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
```

* Uses `SimpleImputer` to impute missing entries in columns 1 and 2 in `x`.
* Missing values (`NaN`) are filled by the mean of their column.

---

## **3. Encoding Categorical Independent Variables**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
```

* The first column (typically containing categorical text data such as country names) is encoded using one-hot encoding.
* This transforms the categorical column into multiple binary columns.
* `remainder='passthrough'` does not change the remaining columns.

---

## **4. Encoding the Dependent Variable**

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

* Converts the target labels ("Yes"/"No") to integers (1/0).

---

## **5. Dividing dataset into Training and Testing**

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```

* Divides the dataset into training and testing:

  * 80% for training
  * 20% for testing
* `random_state=1` enables the same splitting.

---

## **6. Output**

```python
print(x_test)
print(x_train)
print(y_test)
print(y_train)
```

* Prints the processed and divided data arrays to the console.

 
