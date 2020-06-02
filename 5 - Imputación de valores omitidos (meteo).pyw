import sys
import pandas
import pandas as pd
import csv
import numpy as np

#Carga de registros meteorológicos
#Requiere preprocesamiento (compatible con Excel) entre la descarga y su carga en CSV.
chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\API\\AEMET\\reportesAEMET(reducido).csv', chunksize=2000000, low_memory=False,sep=';'):
    chunks.append(chunk)
meteo = pd.concat(chunks, axis=0)

#Cambio de separador decimal, comas por puntos y supresión código 'Ip' en conjunto
meteo['meteo_tmed'] = meteo['meteo_tmed'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_prec'] = meteo['meteo_prec'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_tmin'] = meteo['meteo_tmin'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_tmax'] = meteo['meteo_tmax'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_dir'] = meteo['meteo_dir'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_velmedia'] = meteo['meteo_velmedia'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_racha'] = meteo['meteo_racha'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_presMax'] = meteo['meteo_presMax'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_presMin'] = meteo['meteo_presMin'].astype(str).str.replace(',', ".",regex=True)

meteo['meteo_tmed'] = meteo['meteo_tmed'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_prec'] = meteo['meteo_prec'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_tmin'] = meteo['meteo_tmin'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_tmax'] = meteo['meteo_tmax'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_dir'] = meteo['meteo_dir'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_velmedia'] = meteo['meteo_velmedia'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_racha'] = meteo['meteo_racha'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_presMax'] = meteo['meteo_presMax'].astype(str).str.replace('Ip', '',regex=True).astype(float)
meteo['meteo_presMin'] = meteo['meteo_presMin'].astype(str).str.replace('Ip', '',regex=True).astype(float)

meteo[meteo['meteo_tmin'].isnull()] #Se obtienen índices 141 y 154

#Registros con índice 141 y 154:
#Estrategia de imputación: t_max=(tmax_dia-anterior+tmax_dia-posterior)/2  t_min=(tmin_dia-anterior+tmin_dia-posterior)/2  tmedia=(tmax-tmin)/2
meteo['meteo_tmax'][141]=(meteo['meteo_tmax'][141-1]+meteo['meteo_tmax'][141+1])/2
meteo['meteo_tmin'][141]=(meteo['meteo_tmin'][141-1]+meteo['meteo_tmin'][141+1])/2
meteo['meteo_tmed'][141]=(meteo['meteo_tmax'][141]+meteo['meteo_tmin'][141])/2

meteo['meteo_tmax'][154]=(meteo['meteo_tmax'][154-1]+meteo['meteo_tmax'][154+1])/2
meteo['meteo_tmin'][154]=(meteo['meteo_tmin'][154-1]+meteo['meteo_tmin'][154+1])/2
meteo['meteo_tmed'][154]=(meteo['meteo_tmax'][154]+meteo['meteo_tmin'][154])/2
meteo.info()


#Imputación de valores nulos por variables
#Estrategia de imputación: regresión lineal de valores nulos a partir de otras valores 
from sklearn.linear_model import LinearRegression
regression = LinearRegression()

#Creación de variables día, mes y año.
for i in range(0,len(meteo),1):
    meteo['day'][i]=meteo['meteo_date'][i][0:2] #año
    meteo['month'][i]=meteo['meteo_date'][i][3:5] #mes
    meteo['year'][i]=meteo['meteo_date'][i][6:10] #día

meteo['day']=meteo['day'].astype(int)
meteo['month']=meteo['month'].astype(int)
meteo['year']=meteo['year'].astype(int)

#Regresión 1: Pmax a partir de  temperaturas y fechas
#División en conjunto de entrenamiento (X_train y variable objetivo Y_train) y de evaluación (X_test como insumo e Y_test como resultado)
x_train = meteo[meteo['meteo_presMax'].notnull()].drop(['meteo_date','meteo_prec','meteo_dir','meteo_velmedia','meteo_racha', 'meteo_racha','meteo_presMax','meteo_presMin'], axis=1)
y_train = meteo[meteo['meteo_presMax'].notnull()]['meteo_presMax']
x_test = meteo[meteo['meteo_presMax'].isnull()].drop(['meteo_date','meteo_prec','meteo_dir','meteo_velmedia','meteo_racha', 'meteo_racha','meteo_presMax','meteo_presMin'], axis=1 )
y_test = meteo[meteo['meteo_presMax'].isnull()]['meteo_presMax']
#Entrenamiento del modelo de regresión lineal
regression=LinearRegression().fit(x_train, y_train)
#Predicción del modelo a partir de datos de entrada X_test
predicted = regression.predict(x_test)
#Imputación de valores predichos a los valores nulos  de la variable objetivo del conjunto de datos
meteo.meteo_presMax[meteo.meteo_presMax.isnull()] = predicted


#Regresión 2: Pmin a partir de  temperaturas, fechas y Pmax
#División en conjunto de entrenamiento (X_train y variable objetivo Y_train) y de evaluación (X_test como insumo e Y_test como resultado)
x_train = meteo[meteo['meteo_presMin'].notnull()].drop(['meteo_date','meteo_prec','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMin'], axis=1)
y_train = meteo[meteo['meteo_presMin'].notnull()]['meteo_presMin']
x_test = meteo[meteo['meteo_presMin'].isnull()].drop(['meteo_date','meteo_prec','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMin'], axis=1)
y_test = meteo[meteo['meteo_presMin'].isnull()]['meteo_presMin']
#Entrenamiento del modelo de regresión lineal
regression=LinearRegression().fit(x_train, y_train)
#Predicción del modelo a partir de datos de entrada X_test
predicted = regression.predict(x_test)
#Imputación de valores predichos a los valores nulos  de la variable objetivo del conjunto de datos
meteo.meteo_presMin[meteo.meteo_presMin.isnull()] = predicted
difPres_medio=(meteo['meteo_presMax']-meteo['meteo_presMin']).mean() 
    for i in range(0,len(meteo),1):
     if (meteo['meteo_presMax'][i]-meteo['meteo_presMin'][i]) < 0:
        meteo['meteo_presMin'][i]=(meteo['meteo_presMax'][i]-difPres_medio)


#Regresión 3: Precipitaciones a partir de  temperaturas, fechas, Pmax y Pmin
meteo[meteo['meteo_prec'].isnull()]
#Precipitación sin registros de viento nulos:
#Los valores nulos de precipitación no coinciden, en dichos registros,con valores nulos en el resto de campos. Se predice precipitaciones entrenando con valores de viento no nulos.
#División en conjunto de entrenamiento (X_train y variable objetivo Y_train) y de evaluación (X_test como insumo e Y_test como resultado)
x_train = meteo[meteo['meteo_prec'].notnull() & meteo['meteo_dir'].notnull() & meteo['meteo_velmedia'].notnull() & meteo['meteo_racha'].notnull()].drop(['meteo_date','meteo_prec'], axis=1)
y_train = meteo[meteo['meteo_prec'].notnull() & meteo['meteo_dir'].notnull() & meteo['meteo_velmedia'].notnull() & meteo['meteo_racha'].notnull()]['meteo_prec']
x_test = meteo[meteo['meteo_prec'].isnull()].drop(['meteo_date','meteo_prec'], axis=1)
y_test = meteo[meteo['meteo_prec'].isnull()]['meteo_prec']
#Entrenamiento del modelo de regresión lineal
regression=LinearRegression().fit(x_train, y_train)
#Predicción del modelo a partir de datos de entrada X_test
predicted = regression.predict(x_test)
#Imputación de valores predichos a los valores nulos  de la variable objetivo del conjunto de datos
meteo.meteo_prec[meteo.meteo_prec.isnull()] = predicted

#Regresión 4: Vel media a partir de variables anteriores
#División en conjunto de entrenamiento (X_train y variable objetivo Y_train) y de evaluación (X_test como insumo e Y_test como resultado)
x_train = meteo[meteo['meteo_velmedia'].notnull()].drop(['meteo_date','meteo_dir','meteo_racha','meteo_velmedia'], axis=1)
y_train = meteo[meteo['meteo_velmedia'].notnull()]['meteo_velmedia']
x_test = meteo[meteo['meteo_velmedia'].isnull()].drop(['meteo_date','meteo_dir','meteo_racha','meteo_velmedia'], axis=1)
y_test = meteo[meteo['meteo_velmedia'].isnull()]['meteo_velmedia']
#Entrenamiento del modelo de regresión lineal
regression=LinearRegression().fit(x_train, y_train)
#Predicción del modelo a partir de datos de entrada X_test
predicted = regression.predict(x_test)
#Imputación de valores predichos a los valores nulos  de la variable objetivo del conjunto de datos
meteo.meteo_velmedia[meteo.meteo_velmedia.isnull()] = predicted

#Regresión 5: Dirección del viento a partir de variables anteriores
#División en conjunto de entrenamiento (X_train y variable objetivo Y_train) y de evaluación (X_test como insumo e Y_test como resultado)
x_train = meteo[meteo['meteo_dir'].notnull()].drop(['meteo_date','meteo_dir','meteo_racha'], axis=1)
y_train = meteo[meteo['meteo_dir'].notnull()]['meteo_dir']
x_test = meteo[meteo['meteo_dir'].isnull()].drop(['meteo_date','meteo_dir','meteo_racha'], axis=1)
y_test = meteo[meteo['meteo_dir'].isnull()]['meteo_dir']
#Entrenamiento del modelo de regresión lineal
regression=LinearRegression().fit(x_train, y_train)
#Predicción del modelo a partir de datos de entrada X_test
predicted = regression.predict(x_test)
#Imputación de valores predichos a los valores nulos  de la variable objetivo del conjunto de datos
meteo.meteo_dir[meteo.meteo_dir.isnull()] = predicted
meteo.info()

#Regresión 6: Racha de viento a partir de variables anteriores
#División en conjunto de entrenamiento (X_train y variable objetivo Y_train) y de evaluación (X_test como insumo e Y_test como resultado)
x_train = meteo[meteo['meteo_racha'].notnull()].drop(['meteo_date','meteo_racha'], axis=1)
y_train = meteo[meteo['meteo_racha'].notnull()]['meteo_racha']
x_test = meteo[meteo['meteo_racha'].isnull()].drop(['meteo_date','meteo_racha'], axis=1)
y_test = meteo[meteo['meteo_racha'].isnull()]['meteo_racha']
#Entrenamiento del modelo de regresión lineal
regression=LinearRegression().fit(x_train, y_train)
#Predicción del modelo a partir de datos de entrada X_test
predicted = regression.predict(x_test)
#Imputación de valores predichos a los valores nulos  de la variable objetivo del conjunto de datos
meteo.meteo_racha[meteo.meteo_racha.isnull()] = predicted

#Visibilización del resultado obtenido
meteo.info()


#Almacenamiento local (formato CSV) de registros meteorológicos sin valores nulos
meteo.to_csv('C:\\MYPATH\\TFG\\API\\AEMET\\reportesAEMET(sin missingdata).csv', encoding='utf-8', index=False)


#Asociación de registros subsanados en conjuntos de desplazamientos

#Carga del conjunto datos global, previo a la mitigación de valores nulos.
chunks = []
 for chunk in pd.read_csv('D:\\CS PC MAXI\\Documentos\\ETSII\\TFG\\Datos\\DataSet\\DatosGlobalTrymeteo2.csv', chunksize=20000, low_memory=False,sep=','):
    chunks.append(chunk)
df = pd.concat(chunks, axis=0)
df2=df

#Comprobación de valores nulos existentes
for i in df2.columns:
    print(i+':'+ str(len(df2[df2[i].isnull()])))

#Supresión de variables metereológicas de registros de desplazamiento y nueva asociación a registros meteorológicos subsanados
meteo.info()
df3=df2.drop(['meteo_tmed','meteo_prec','meteo_tmin','meteo_tmax','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMax','meteo_presMin'], axis=1)
meteo['meteo_date']=meteo['day'].map(str)+'/'+meteo['month'].map(str)+'/'+meteo['year'].map(str)
df4=pd.merge(df3,meteo)
len(df3)
df4=df4.drop_duplicates()
len(df4)  
df4.head()

#Comprobación resultado
for i in df4.columns:
    print(i+':'+ str(len(df4[df4[i].isnull()])))

#Supresión de variable 'zip_code'
df4=df4.drop(['zip_code'],axis=1)

#Almacenamiento local
df4.to_csv('C:\\MYPATH\\TFG\\Datos\\DataSet\\parcial_trayectosBiciMAD_v2.csv', encoding='utf-8', index=False, sep=';')









