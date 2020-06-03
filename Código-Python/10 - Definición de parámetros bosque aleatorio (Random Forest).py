import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import time
start_time = time.time()


pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 810)

#Importación de conjunto de datos de la demanda del servicio BiciMAD
chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\Datos\\DataSet\\demandaBiciMAD_v1.csv', chunksize=200, low_memory=False,dtype={'_id':str,'ageRange':int, 'idplug_base':str, 'idplug_station' : str, 'idunplug_base': str, 'idunplug_station': str, 'travel_time': int, 'unplug_hourTime': str, 'user_day_code': str, 'user_type':str , 'zip_code':str},sep=','):
    chunks.append(chunk)
df5 = pd.concat(chunks, axis=0)

df5=pd.DataFrame(df5)
df5=df5.sort_values(by='datetime')
df5['cont']=df5.index.astype(int)

#Características que contribuyen
X=df5[['year','month','day','weekday_int','meteo_tmed','meteo_prec','meteo_tmin','meteo_tmax','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMax','meteo_presMin','cont']].astype(float).values
df5x=df5[['year','month','day','weekday_int','meteo_tmed','meteo_prec','meteo_tmin','meteo_tmax','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMax','meteo_presMin','cont']]
#Variable a predecir (desplazamientos por día)
y=df5['_id'].astype(int).values

#División en 60% entrenamiento 20% evaluación 20% validación
X, X_eval, y, y_eval = train_test_split(X, y, test_size=0.2, random_state=1)
# train 485 registros,  test 162 registros , eval 162 registros

#Se han empleado líneas de código procedentes del recurso https://medium.com/@nilimeshhalder/how-to-find-optimal-parameters-for-catboost-using-gridsearchcv-for-regression-in-python-ef778b60d95d
m_time = time.time()
print("GridSearch:--- %s segundos ---" % (m_time - start_time))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
modelo =RandomForestRegressor(max_depth=None, warm_start=True)

#Declaración de valores que se dean probar en diferentes hiper-parámetros
parametros = {'min_samples_leaf': [2,3,4,5,6],
                  'criterion' : ['mse','mae'],
                  'oob_score' : ['False','True'],
                  'bootstrap' : ['False','True'],
                  'max_features' : ['auto','sqrt', 7, 10],
                  'min_impurity_decrease' : [0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
                  'n_estimators' : [20,100,1000,2000,5000],
                  'ccp_alpha' : [0.1, None]}

#Búsqueda secuencial emplendo validación cruzada k=5 
    grid = GridSearchCV(estimator=modelo, param_grid = parametros, cv = 5, n_jobs=-1, verbose=2)
#Advertencia: la siguiente línea de código conlleva la ejecución de un proceso que puede tomar varias horas. La configuración del modelo con los hiperparámetros resultantes se encuentra más adelante.
    grid.fit(X, y)    

    # Representación de resultados obtenidos
    print("\n========================================================")
    print(" Resultado de búsqueda por Grid" )
    print("========================================================")    
    
    print("\n Mejor configuración del estimador entre todos los parámetros ensayados:\n",
          grid.best_estimator_)
    
    print("\n El mejor índice de regresión ensayado:\n",
          grid.best_score_)
    
    print("\n Mejores parámetros entre los buscados:\n",
          grid.best_params_)
    
    print("\n ========================================================")
    
    print("GridSearch:--- %s segundos ---" % (time.time() - m_time))
    print("Total:--- %s segundos ---" % (time.time() - start_time))


#Definición de hiperparámetros, resultante de la búsqueda secuencial, para modelo ExtraTree
   '''   modelo= RandomForestRegressor(bootstrap='False', ccp_alpha=0.1, criterion='mse',
                      max_depth=None, max_features=10, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.3,
                      min_impurity_split=None, min_samples_leaf=3,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score='False',
                      random_state=None, verbose=0, warm_start=True)'''




