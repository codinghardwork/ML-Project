## Data Preprocessing with Python and scikit-learn

This script demonstrates essential steps for preparing a dataset for machine learning using Python libraries such as NumPy, Pandas, and scikit-learn. It includes handling missing data, encoding categorical variables, and splitting the dataset into training and testing sets.

### Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

* **NumPy**: For numerical operations.
* **Matplotlib**: Included for potential visualizations (not used in this script).
* **Pandas**: For reading and manipulating data in tabular form.

---

### 1. Importing the Dataset

```python
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

* Loads the data from a CSV file named `Data.csv`.
* `x` contains all the input features (independent variables).
* `y` contains the target variable (dependent variable).

---

### 2. Handling Missing Data

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
```

* Uses `SimpleImputer` to fill in missing values in columns 1 and 2 of `x`.
* Missing values (`NaN`) are replaced with the **mean** of their respective columns.

---

### 3. Encoding Categorical Data (Independent Variables)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
```

* The first column (typically containing categorical text data such as country names) is encoded using one-hot encoding.
* This transforms the categorical column into multiple binary columns.
* `remainder='passthrough'` ensures the rest of the columns are left unchanged.

---

### 4. Encoding the Dependent Variable

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

* Encodes the target labels (e.g., "Yes"/"No") as integers (e.g., 1/0).

---

### 5. Splitting the Dataset into Training and Testing Sets

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```

* Splits the dataset into training and testing sets:

  * 80% for training
  * 20% for testing
* `random_state=1` ensures that the split is reproducible.

---

### 6. Output

```python
print(x_test)
print(x_train)
print(y_test)
print(y_train)
```

* Prints the processed and split data arrays to verify correctness.

