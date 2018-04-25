import numpy as np
import pandas as pd
import math
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_val_predict
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import SelectFromModel
import ml_metrics as metrics
from sklearn.grid_search import GridSearchCV
import gc

# Set seed to achieve reproducible result
SEED = 450
np.random.seed(SEED)

# Import prepared data
df = pd.read_csv('df_after_lag_feateng.csv')
temp = pd.read_csv('san-col3merge_nonbalanced.csv')

y_colheaders = temp.columns[-24:]
y = df.loc[:,y_colheaders]
X = df.drop(y_colheaders, axis=1)

temp = []

# Shuffle data rows
randomise_vector = np.random.permutation(len(X))
X = X.iloc[randomise_vector]
y = y.iloc[randomise_vector]
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

## Prepare data for compatibility with the Python Random Forest library
# Convert categorical columns to dummy columns of binary values
categorical_cols = ["indrel", "ind_empleado", "pais_residencia", "sexo", "tiprel_1mes", "indrel_1mes", "indresi", "indext", "conyuemp", "canal_entrada", "indfall", "nomprov", "segmento"]
X = pd.get_dummies(X, columns = categorical_cols)

# Drop date columns
X = X.drop("index", 1)
X = X.drop("fecha_dato", 1)
X = X.drop("fecha_alta", 1)
X = X.drop("ult_fec_cli_1t", 1)
X = X.drop("primary_customer_time", 1) # don't drop column if using decision trees

# Split into train & test data
rows = len(X)
Xtest = X.iloc[:int(rows*0.8),:]
X = X.iloc[int(rows*0.8):,:]
ytest = y.iloc[:int(rows*0.8),:]
y = y.iloc[int(rows*0.8):,:]
print(X.shape, Xtest.shape, y.shape, ytest.shape)

# Convert pandas dataframe to numpy array
y = y.values
ytest = ytest.values


## FEATURE SELECTION - RANDOM FOREST
# Initial model fitting
rfc = RandomForestClassifier()
rfc.fit(X, y)

# Feature selection
selector = SelectFromModel(rfc, threshold='1.25*mean')
X_reduced = selector.fit_transform(X, y)
print(X.shape)
print(X_reduced.shape)

%%time
## Grid search method
param_grid = {"max_features":[50,80,109], "min_samples_split":[2,4,8], "min_samples_leaf":[1,3], "bootstrap":[True,False]}
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=3)
grid_search.fit(X, y)

print(grid_search.best_params_)


## MODEL FITTING - RANDOM FOREST (optimised hyperparameters)
rfc_optim = RandomForestClassifier(bootstrap=False, criterion="gini", min_samples_split=4, min_samples_leaf=1, max_depth=None, max_features=109)
rfc_optim.fit(X, y)


## PREDICTION - RANDOM FOREST - training set
prediction_cv = cross_val_predict(rfc_optim, X, y, cv=5)


## ACCURACY - RANDOM FOREST - training set
train_accuracy = sum(np.sum(prediction_cv - y, axis=1)==0)/len(y)
print('Training accuracy: ' + str(train_accuracy*100) + '%')

# Un-binarise output matrix
prediction_cv_unbin = []
for iter in range(len(prediction_cv)):
    prediction_cv_unbin.append(list(np.where(prediction_cv[iter]==1)[0]))

y_unbin = []
for iter in range(len(y)):
    y_unbin.append(list(np.where(y[iter]==1)[0]))

# Mean Average Precision @7
print('MAP@7: ' + str(metrics.mapk(y_unbin, prediction_cv_unbin, 7)))


## PREDICTION - RANDOM FOREST - test set
prediction_test = rfc_optim.predict(Xtest)
np.savetxt("prediction-rforest-test.csv", prediction_test, delimiter=",")


## ACCURACY - RANDOM FOREST - test set
train_accuracy = sum(np.sum(prediction_test - ytest, axis=1)==0)/len(ytest)
print('Training accuracy: ' + str(train_accuracy*100) + '%')

# Un-binarise output matrix
prediction_test_unbin = []
for iter in range(len(prediction_test)):
    prediction_test_unbin.append(list(np.where(prediction_test[iter]==1)[0]))

ytest_unbin = []
for iter in range(len(ytest)):
    ytest_unbin.append(list(np.where(ytest[iter]==1)[0]))

# Mean Average Precision @7
print('MAP@7: ' + str(metrics.mapk(ytest_unbin, prediction_test_unbin, 7)))
