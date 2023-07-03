# Medical Insurance Prediction

This project aim to establish a simple prediction model for medical insurance via mechine learning.<br>

## Model choosing

The models used in this project include the following:<br>

* [LinearRegression](https://www.ibm.com/topics/linear-regression)
* [PolynomialFeatures](https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/)
* [RandomForest](https://www.ibm.com/topics/random-forest)
### LinearRegression 

```py
from sklearn.linear_model import LinearRegression
```

`Linear regression` is suitable for data with:<br>
* Continuous target variable
* Linear relationship
* Independent features
* Normally distributed error term

### PolynomialFeatures 

```py
from sklearn.preprocessing import PolynomialFeatures
```

`PolynomialFeatures` is suitable for data with:<br>
* Interaction effects
* Higher-order relationships
* Non-linear relationships

### RandomForest

```py
from sklearn.ensemble import RandomForestRegressor
```

`RandomForest` is suitable for data with:<be>
* Outliers and missing values
* High-dimensional data
* Non-linear relationships 

And `RandomForest` model has following funtion:

*Feature importance* : Provides a measure of feature importance, which can help identify the most influential features for the target variable.<br>
*Ensemble learning* : Combines multiple decision trees to make predictions.<br>

