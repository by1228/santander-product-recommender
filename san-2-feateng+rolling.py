import numpy as np
import pandas as pd
import math
from datetime import timedelta
from sklearn.model_selection import StratifiedShuffleSplit
import time
import gc

t0 = time.clock()

# Set seed to achieve reproducible result
SEED = 450
np.random.seed(SEED)

# Import balancedextract
df = pd.read_csv('san-col3merge_nonbalanced.csv', dtype={"sexo":str, "ind_nuevo":str, "ult_fec_cli_1t":str, "indext":str, "conyuemp":str})
df = df.iloc[np.random.permutation(len(df))] # shuffle data rows
df.reset_index(drop=True, inplace=True)
df = df.drop('Unnamed: 0', 1)
df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A" # Fix the special character in A Coruna
# Save a set of test data for post-modelling evaluation
X = df.iloc[:int(len(df)*0.8), :24]
y = df.iloc[:int(len(df)*0.8), 24:]
Xtest = df.iloc[int(len(df)*0.8):, :24]
ytest = df.iloc[int(len(df)*0.8):, 24:]

## Data cleaning
# Convert age
X["age"]   = pd.to_numeric(X["age"], errors="coerce")
Xtest["age"]   = pd.to_numeric(Xtest["age"], errors="coerce")
# Age is bimodal, concentrated at 25 and 40. Separate the distribution and move the outliers to the mean of the closest group.
mean_closest_group = X.loc[(X.age >= 18) & (X.age <= 30),"age"].mean(skipna=True)
X.loc[X.age < 18,"age"] = mean_closest_group
Xtest.loc[Xtest.age < 18,"age"] = mean_closest_group
X.loc[X.age > 100,"age"] = mean_closest_group
Xtest.loc[Xtest.age > 100,"age"] = mean_closest_group
mean_rest = X["age"].mean()
X["age"].fillna(mean_rest,inplace=True)
Xtest["age"].fillna(mean_rest,inplace=True)
X["age"] = X["age"].astype(int)
Xtest["age"] = Xtest["age"].astype(int)

# Check missing entries in columns ind_nuevo, antiguedad, ind_actividad_cliente etc. and will see all the entries are common to all columns --> bad entries, just remove them
notnullrows = X["ind_nuevo"].notnull()
X = X.loc[notnullrows, :]
y = y.loc[notnullrows, :]
notnullrowstest = Xtest["ind_nuevo"].notnull()
Xtest = Xtest.loc[notnullrowstest, :]
ytest = ytest.loc[notnullrowstest, :]

# Impute missing dates with median
dates = X.loc[:,"fecha_alta"].sort_values().reset_index()
median_date = int(np.median(dates.index.values))
X.loc[X.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]
Xtest.loc[Xtest.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

# Convert indrel to integer
X.loc[X.indrel.notnull(),"indrel"] = X.loc[X.indrel.notnull(),"indrel"].astype(int)
Xtest.loc[Xtest.indrel.notnull(),"indrel"] = Xtest.loc[Xtest.indrel.notnull(),"indrel"].astype(int)

# Convert dates
X["fecha_dato"] = pd.to_datetime(X["fecha_dato"],format="%Y-%m-%d")
# X["fecha_dato_day"] = X["fecha_dato"].map(lambda x: x.day)
# X["fecha_dato_month"] = X["fecha_dato"].map(lambda x: x.month)
# X["fecha_dato_year"] = X["fecha_dato"].map(lambda x: x.year)
X["fecha_alta"] = pd.to_datetime(X["fecha_alta"],format="%Y-%m-%d")
# X["fecha_alta_day"] = X["fecha_alta"].map(lambda x: x.day)
# X["fecha_alta_month"] = X["fecha_alta"].map(lambda x: x.month)
# X["fecha_alta_year"] = X["fecha_alta"].map(lambda x: x.year)
X["ult_fec_cli_1t"] = pd.to_datetime(X["ult_fec_cli_1t"],format="%Y-%m-%d")
Xtest["fecha_dato"] = pd.to_datetime(Xtest["fecha_dato"],format="%Y-%m-%d")
Xtest["fecha_alta"] = pd.to_datetime(Xtest["fecha_alta"],format="%Y-%m-%d")
Xtest["ult_fec_cli_1t"] = pd.to_datetime(Xtest["ult_fec_cli_1t"],format="%Y-%m-%d")

# tipodom not useful; province code not needed since name of province exists in nomprov; drop both
X.drop(["tipodom","cod_prov"],axis=1,inplace=True)
Xtest.drop(["tipodom","cod_prov"],axis=1,inplace=True)


# Rename missing city
X.loc[X.nomprov.isnull(),"nomprov"] = "UNKNOWN"
Xtest.loc[Xtest.nomprov.isnull(),"nomprov"] = "UNKNOWN"

# Assign missing incomes by province
# Rationale: Step 1 - fill null cells with province median (grouped); Step 2 - fill remaining cells with overall median (if a providence median is missing in grouped)
grouped = X.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()

new_incomes = pd.merge(X, grouped, how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
new_incomes = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
X.sort_values("nomprov",inplace=True)
X = X.reset_index()
new_incomes = new_incomes.reset_index() 

X.loc[X.renta.isnull(),"renta"] = new_incomes.loc[X.renta.isnull(),"renta"].reset_index() # Step 1
X.loc[X.renta.isnull(),"renta"] = X.loc[X.renta.notnull(),"renta"].median() # Step 2

# Same for Xtest
new_incomes_test = pd.merge(Xtest, grouped, how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
new_incomes_test = new_incomes_test.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
Xtest.sort_values("nomprov",inplace=True)
Xtest = Xtest.reset_index()
new_incomes_test = new_incomes_test.reset_index()

Xtest.loc[Xtest.renta.isnull(),"renta"] = new_incomes_test.loc[Xtest.renta.isnull(),"renta"].reset_index()
Xtest.loc[Xtest.renta.isnull(),"renta"] = Xtest.loc[Xtest.renta.notnull(),"renta"].median()


# Replace ind_nomina_ult1 and ind_nom_pens_ult1 by 0 since there are few missing entries; ideally should look at previous month
y.loc[y.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
y.loc[y.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
y["ind_nomina_ult1"] = y["ind_nomina_ult1"].astype(int)
y["ind_nom_pens_ult1"] = y["ind_nom_pens_ult1"].astype(int)

ytest.loc[ytest.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
ytest.loc[ytest.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0
ytest["ind_nomina_ult1"] = ytest["ind_nomina_ult1"].astype(int)
ytest["ind_nom_pens_ult1"] = ytest["ind_nom_pens_ult1"].astype(int)


# Fill empty strings either with the most common value or create an unknown category
string_data = X.select_dtypes(include=["object"])
missing_columns = [col for col in string_data if string_data[col].isnull().any()]
del string_data

X.loc[X.indfall.isnull(),"indfall"] = "N"
X.loc[X.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
X.tiprel_1mes = X.tiprel_1mes.astype("category")

map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "P",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}

X.indrel_1mes.fillna("P",inplace=True)
X.indrel_1mes = X.indrel_1mes.apply(lambda x: map_dict.get(x,x))
X.indrel_1mes = X.indrel_1mes.astype("category")

unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    X.loc[X[col].isnull(),col] = "UNKNOWN"

# Same for Xtest
string_data_test = Xtest.select_dtypes(include=["object"])
missing_columns_test = [col for col in string_data_test if string_data_test[col].isnull().any()]
del string_data_test

Xtest.loc[Xtest.indfall.isnull(),"indfall"] = "N"
Xtest.loc[Xtest.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
Xtest.tiprel_1mes = Xtest.tiprel_1mes.astype("category")

Xtest.indrel_1mes.fillna("P",inplace=True)
Xtest.indrel_1mes = Xtest.indrel_1mes.apply(lambda x: map_dict.get(x,x))
Xtest.indrel_1mes = Xtest.indrel_1mes.astype("category")

unknown_cols_test = [col for col in missing_columns_test if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols_test:
    Xtest.loc[Xtest[col].isnull(),col] = "UNKNOWN"
	

## Feature engineering
# Add feature: time interval of a customer being a primary customer since joining
X.loc[X["ult_fec_cli_1t"].notnull(), "primary_customer_time"] = ((X.loc[X["ult_fec_cli_1t"].notnull(),"ult_fec_cli_1t"] - X.loc[X["ult_fec_cli_1t"].notnull(),"fecha_alta"])/np.timedelta64(1, 'D')).astype(int)
Xtest.loc[Xtest["ult_fec_cli_1t"].notnull(), "primary_customer_time"] = ((Xtest.loc[Xtest["ult_fec_cli_1t"].notnull(),"ult_fec_cli_1t"] - Xtest.loc[Xtest["ult_fec_cli_1t"].notnull(),"fecha_alta"])/np.timedelta64(1, 'D')).astype(int)

df = pd.concat([X,y], axis=1)
df_test = pd.concat([Xtest,ytest], axis=1)


# Add feature: lagged variables - from t=end of period and rolling t
end_of_period = pd.to_datetime('2016-05-28',format="%Y-%m-%d")

all_prods = []
all_prods_rolling = []
for iter in list(y.columns):
    for jter in range(18):
        all_prods.append("lag"+str(jter)+str(iter))
        all_prods_rolling.append("lagrolling"+str(jter)+str(iter))

gc.collect()
print(df.shape)

df = pd.concat([df, pd.DataFrame(columns=all_prods), pd.DataFrame(columns=all_prods_rolling)]) # Create columns for lag features

#df.to_csv('df_before_lag_feateng.csv', sep=',', encoding='utf-8', index=False)

df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")

df_grouped = df.groupby('ncodpers')


df_lag = df.copy()

for key, item in df_grouped:
    item = item.sort_values('fecha_dato') # order each ncodper by ascending facho_dato date
    lag = ((end_of_period - item.fecha_dato)/np.timedelta64(1, 'M')).astype(int) # get number of months lagged for the record entry
    idx = item.index
    bt = df.loc[idx,list(y.columns)].apply(lambda x: x>0) # get booleans for whether each product was purchased
    prod = bt.apply(lambda x: list(y.columns[x.values]), axis=1) # extract names of products purchased - FOR EACH ncodper within each ncodper group
    
    for iter in range(len(idx)): # iterate through each entry within an ncodper group, indexed by idx (index number of each entry, unordered)     
        # Lagged variables from end of observation period (1.5 years)
        lagcol = ["lag"+str(lag[idx[iter]])+str(kter) for kter in prod[idx[iter]]] 
        df_lag.loc[idx[iter], lagcol] = 1 # Mark lagged features from t=end of period as 1 for purchased products
        
        # Add rolling feature
        if iter>=1:
            for jter in range(iter):
                roll_diff = int((item.loc[idx[iter],'fecha_dato'] - item.loc[idx[jter],'fecha_dato']).days/30) # Date difference (in months) between date of purchase in an entry and previous entry(ies) in the same ncodper group
                roll_bt = df.loc[idx[jter],list(y.columns)].apply(lambda x: x>0) # get booleans for whether each product was purchased for each previous entry in the same ncodper group
                roll_prod = list(y.columns[roll_bt.values]) # extract names of products purchased for each previous entry in the same ncodper group
                lagrollcol = ["lagrolling"+str(roll_diff)+str(kter) for kter in roll_prod]
                df_lag.loc[idx[iter], lagrollcol] = 1 # Mark rolling lagged features as 1 for purchased products
    
                
df = df_lag.fillna(value=0)


df.to_csv('df_after_lag_feateng.csv', sep=',', encoding='utf-8', index=False)


t1 = time.clock()
print('Time elapsed:' + str(t1-t0) + 'seconds')