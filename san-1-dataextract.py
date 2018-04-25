import numpy as np
import pandas as pd
import random
import gc

df = pd.read_csv('train_ver2.csv', dtype={"fecha_dato":str, "ncodpers":str, "ind_empleado":str, "pais_residencia":str, "sexo":str, "age":str, "fecha_alta":str, "ind_nuevo":str, "antiguedad":str, "indrel":str, "ult_fec_cli_1t":str, "indrel_1mes":str, "tiprel_1mes":str, "indresi":str, "indext":str, "ult_fec_cli_1t":str, "indext":str, "conyuemp":str, "canal_entrada":str, "indfall":str, "tipodom":str, "cod_prov":str, "nomprov":str, "ind_actividad_cliente":str, "renta":float, "segmento":str})

df = df.iloc[np.random.permutation(len(df))]

# Split data on column 3 of target variable (ind_cco_fin_ult1) =0
# Doing in chunks to avoid running out of memory
col30 = pd.DataFrame()
for iter in range(10):
    temp = df.iloc[int(len(df)/10*iter):int(len(df)/10*(iter+1)),]
    temp = temp.iloc[np.array(temp.ind_cco_fin_ult1==0),]
    col30 = col30.append(temp)
    gc.collect() # dump cache
print(col30.shape)

# Split data on column 3 of target variable (ind_cco_fin_ult1) =1
# Doing in chunks to avoid running out of memory
col31 = pd.DataFrame()
for iter in range(50):
    temp = df.iloc[int(len(df)/50*iter):int(len(df)/50*(iter+1)),]
    temp = temp.iloc[np.array(temp.ind_cco_fin_ult1==1),]
    col31 = col31.append(temp)
    gc.collect() # dump cache
print(col31.shape)






col30extract = pd.DataFrame()
for iter in range(24,48):
    temp = col30[np.array(col30.iloc[:,iter].apply(lambda x: x > 0))]
    temp = temp.iloc[:180000,]
    col30extract = col30extract.append(temp)
    
# Remove duplicate rows
col30extract = col30extract.drop_duplicates()

# Shuffle data
col30extract = col30extract.iloc[np.random.permutation(len(col30extract))]
col30extract.reset_index(drop=True, inplace=True)

# Output
col30extract.to_csv('san-col30.csv', sep=',', encoding='utf-8')
print(len(col30extract))





col31extract = pd.DataFrame()
for iter in range(24,48):
    temp = col31[np.array(col31.iloc[:,iter].apply(lambda x: x > 0))]
    temp = temp.iloc[:180000,]
    col31extract = col31extract.append(temp)
    
# Remove duplicate rows
col31extract = col31extract.drop_duplicates()

# Shuffle data
col31extract = col31extract.iloc[np.random.permutation(len(col31extract))]
col31extract.reset_index(drop=True, inplace=True)

# Output
col31extract.to_csv('san-col31extract.csv', sep=',', encoding='utf-8')
print(len(col31extract))



# Join the tables with col3 =0 and =1
col3merge = pd.concat([col30extract,col31extract])

# Output
col3merge.to_csv('san-col3merge_nonbalanced.csv', sep=',', encoding='utf-8')