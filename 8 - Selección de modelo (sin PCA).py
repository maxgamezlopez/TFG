import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model 
from sklearn.model_selection import cross_val_score

#Importación de conjunto de datos de demanda del servicio BiciMAD
chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\Datos\\DataSet\\demandaBiciMAD_v1.csv', chunksize=200, low_memory=False,sep=','):
    chunks.append(chunk)
df5 = pd.concat(chunks, axis=0)

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 820)

df5['cont']=df5.index.astype(int)

#Características que contribuyen
X=df5[['year','month','day','weekday_int','meteo_tmed','meteo_prec','meteo_tmin','meteo_tmax','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMax','meteo_presMin','cont']].astype(float).values
df5x=df5[['year','month','day','weekday_int','meteo_tmed','meteo_prec','meteo_tmin','meteo_tmax','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMax','meteo_presMin','cont']]
#Variable a predecir (desplazamientos por día)
y=df5['_id'].astype(int).values

from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression,LogisticRegression, RidgeCV, Lasso,ElasticNetCV, BayesianRidge,PassiveAggressiveRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor, GradientBoostingRegressor

#Separación en conjuntos de entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Declaración de diferentes técnicas a ensayar
MODELOS.items()
MODELOS = {
    'Linear regression': LinearRegression(),
    'LogRegression': LogisticRegression(),
    'Ridge': RidgeCV(),
    'BayesianRidge': BayesianRidge(),
    'Lasso': Lasso(normalize=True), 
    'ElasticNet' :ElasticNetCV(cv=5, random_state=0),
    'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
    'Supported Vector Machine Reg' :svm.SVR(),
    'K-nn': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'Random Forest' : RandomForestRegressor(random_state=0, n_estimators=10),
    'Extra trees': ExtraTreesRegressor(max_features=int(13),random_state=0),
    'GPR' : GaussianProcessRegressor(kernel=(DotProduct() + WhiteKernel()), random_state=0),
    'GBR' : GradientBoostingRegressor(random_state=0, n_estimators=10)
}

#Normalización (a escala) de los datos empleados
from sklearn import preprocessing
scale = StandardScaler()
X_train_std = scale.fit_transform(X_train)
X_test_std = scale.transform(X_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import explained_variance_score, r2_score

#Se declaran algunos diccionarios que permiten almacenar datos predichos y estadísticos sobre la calidad del ajusta
y_test_predict = dict()
R = dict()
RR=dict()
 
resultados=pd.DataFrame()
R2=dict()
ExpVar=dict()
for name, estimator in MODELOS.items():
        estimator.fit(X_train_std, y_train)
        y_test_predict[name] = estimator.predict(X_test_std)
        resultados[name]=y_test_predict[name]
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=int(0))
        R2[name]=cross_val_score(estimator, X, y, scoring='r2', cv=cv, n_jobs=-1).mean()
        ExpVar[name]=cross_val_score(estimator, X, y,scoring='explained_variance', cv=cv, n_jobs=-1).mean()
       
        
#Conversión de diccionarios en DataFrame para facilitar su visualización
R2=pd.DataFrame(R2.items()).T
ExpVar=pd.DataFrame(ExpVar.items()).T
Comparativa=pd.concat([R2,ExpVar]).drop_duplicates()
#Visualización del ajuste de cada modelo en términos de coef. determinación y varianza explicada
Comparativa.head()
#En términos de coeficiente de determinación <<<Ilustración 7.15>>>
R2.head()

#Generación de histogramas <<<Ilustración 7.16>>>

resultados['real']=y_test
import matplotlib.pyplot as plt
import numpy as np

#HISTOGRAMA Regresión lineal
resultados['error Linear regression']=(resultados['Linear regression']-resultados['real'])
plt.subplot(231)
plt.title('Regresión lineal')
resultados['error Linear regression'].plot.hist(bins=150, xlim=(-7000,7000), ylim=(0,15),color='#002e5d')
plt.ylabel('Frecuencia')
plt.xlabel('Error')

#HISTOGRAMA Regresión método Lasso
resultados['error Lasso']=(resultados['Lasso']-resultados['real'])
plt.subplot(232)
plt.title('Regresión Lasso')
resultados['error Lasso'].plot.hist(bins=150, xlim=(-7000,7000),ylim=(0,15),color='#002e5d')
plt.ylabel('Frecuencia')
plt.xlabel('Error')

#HISTOGRAMA K-nn
resultados['error K-nn']=(resultados['K-nn']-resultados['real'])
plt.subplot(233)
plt.title('K-vecinos')
resultados['error K-nn'].plot.hist(bins=150, xlim=(-7000,7000),ylim=(0,15),color='#002e5d')
plt.ylabel('Frecuencia')
plt.xlabel('Error')

#HISTOGRAMA Extra Trees
resultados['error Extra trees']=(resultados['Extra trees']-resultados['real'])
plt.subplot(234)
plt.title('Árboles ext. aleatorios')
resultados['error Extra trees'].plot.hist(bins=150, xlim=(-7000,7000),ylim=(0,15),color='#002e5d')
plt.ylabel('Frecuencia')
plt.xlabel('Error')

#HISTOGRAMA Random Forest
resultados['error Random Forest']=(resultados['Random Forest']-resultados['real'])
plt.subplot(235)
plt.title('Bosque aleatorio')
resultados['error Random Forest'].plot.hist(bins=150, xlim=(-7000,7000),ylim=(0,15),color='#002e5d')
plt.ylabel('Frecuencia')
plt.xlabel('Error')

#HISTOGRAMA GBR
resultados['error GBR']=(resultados['GBR']-resultados['real'])
plt.subplot(236)
plt.title('Potenciación del gradiente en árboles')
resultados['error GBR'].plot.hist(bins=150, xlim=(-7000,7000),ylim=(0,15),color='#002e5d')
plt.ylabel('Frecuencia')
plt.xlabel('Error')

plt.tight_layout(pad=0.1)
plt.show()

resultados.head()
#HISTOGRAMA Extra tree
resultados['error PassiveAggressiveRegressor']=(resultados['PassiveAggressiveRegressor']-resultados['real'])
a[1][1]=resultados['error PassiveAggressiveRegressor'].plot.hist(bins=150, xlim=(-7000,7000),color='#002e5d')
plt.title('Histograma error absoluto de PassiveAggressiveRegressor')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
plt.show()

#Cálculo de raíz de error cuadrático de algunos estimadores
resultados['error K-nn']=((resultados['K-nn']-resultados['real'])**2)**(1/2)
resultados['error Linear regression']=((resultados['Linear regression']-resultados['real'])**2)**(1/2)
resultados['error Ridge']=((resultados['Ridge']-resultados['real'])**2)**(1/2)

#Gráfico de valores predichos frente a reales
resultados.plot.scatter(y='Extra trees', x=['real'], figsize=(10,10), grid=True,s=1)
    x = [1,30000]
    y = [1,30000]
plt.plot(x,y, color="red", label = 'línea de referencia: tiempo previsto igual al de desplazamiento', linewidth=1 )
plt.plot([0,1],[0,1])
plt.xlabel('real')
plt.ylabel('predicho')
plt.show()