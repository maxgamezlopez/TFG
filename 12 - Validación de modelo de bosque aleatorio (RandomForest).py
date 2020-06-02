import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import time
start_time = time.time()

pd.set_option('display.max_columns', 17)
pd.set_option('display.max_rows', 810)

#Importación de conjuntos de datos de la demanda de servicio BiciMAD
chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\Datos\\DataSet\\demandaBiciMAD_v1.csv', chunksize=200, low_memory=False,sep=','):
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
X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
# train 485 reg,  test 162 reg , eval 162 reg

#ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score,r2_score

 modelo= RandomForestRegressor(bootstrap='False', ccp_alpha=0.1, criterion='mse',
                      max_depth=None, max_features=10, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.3,
                      min_impurity_split=None, min_samples_leaf=3,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score='False',
                      random_state=None, verbose=0, warm_start=True)

 modelo.fit(X, y)
 y_test_predict = modelo.predict(X_val)
 resultados=y_test_predict

 #Obtención de coeficiente de determinación:
 r2_score(resultados, y_val)

resRF=pd.DataFrame()
resRF['predicho']=resultados
resRF['real']=y_val

resRF['errorAbs']=resRF['predicho']-resRF['real']
resRF['errorCuad']=(resRF['predicho']-resRF['real'])**(2)

#Visualización del resultado mediante histograma de error absoluto  <<< Ilustración 7.31>>>
plt.title('Histograma de errores absolutos (bosque aleatorio)')
resRF['errorAbs'].plot.hist(bins=150, xlim=(-7000,7000), ylim=(0,20),color='#002e5d')
plt.ylabel('Frecuencia')
plt.xlabel('Error absoluto')
resRF['errorAbs'].describe()
plt.show()

#Descripción analítica de error absouto
resRF['errorAbs'].describe()

#Representación de valores predichos frente a reales <<<Ilustración 7.30>>>
resRF.plot.scatter(y='predicho', x=['real'], figsize=(10,10), grid=True,s=1)
    x = [1,30000]
    y = [1,30000]
plt.plot(x,y, color="red", label = 'línea de referencia: tiempo previsto igual al de desplazamiento', linewidth=1 )
plt.title('Valores predichos frente a reales (bosque aleatorio)')
plt.plot([0,1],[0,1])
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.show()

#Almacenamiento del modelo en fichero local
from sklearn.externals import joblib
joblib.dump(modelo, 'C:\\MYPATH\\TFG\\Python\\Modelos\\Modelo Random Forest V1')

#El modelo puede ser cargado y empleado para la predicción de nuevos valores.
modelo = joblib.load('C:\\MYPATH\\TFG\\Python\\Modelos\\Modelo Random Forest V1')




