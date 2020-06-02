import sys
import pandas
import pandas as pd
import csv
import time
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Importación de datos de demanda de BiciMAD

chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\Datos\\DataSet\\demandaBiciMAD_v1.csv', chunksize=200, low_memory=False,sep=','):
    chunks.append(chunk)
df5 = pd.concat(chunks, axis=0)

#Separación en variable objetivo (Y) y atributos (X) (véase Ilustración 7.7)
#Características que contribuyen
x=df5[['year','month','day','weekday_int','meteo_tmed','meteo_prec','meteo_tmin','meteo_tmax','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMax','meteo_presMin']].astype(float).values
df5x=df5[['year','month','day','weekday_int','meteo_tmed','meteo_prec','meteo_tmin','meteo_tmax','meteo_dir','meteo_velmedia','meteo_racha','meteo_presMax','meteo_presMin']]
#Variable a predecir (desplazamientos por día)
y=df5['_id'].astype(int).values

#Líneas de código desarrolladas empleando como referencia el recurso bibliográfico disponible en https://www.aprendemachinelearning.com/comprende-principal-component-analysis/
#Estandarización de características (input)
x = StandardScaler().fit_transform(x)
scaler=StandardScaler()
scaler.fit(x) # Cálculo del valor promedio
X_scaled=scaler.transform(x)# Resto de valores escalados con base en promedio

#Consideración de todas las componentes principales, tantas como atributos
pca=PCA(n_components=13) # Otra opción es instanciar pca sólo con dimensiones nuevas hasta obtene
pca.fit(X_scaled) # obtener los componentes principales
X_pca=pca.transform(X_scaled) # convertimos nuestros datos con las nuevas dimensiones de PCA
print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
#Varianza explicada por todas las componentes principales (100%)
print('suma:',sum(expl[0:13]))

#####################
#Gráfico de cargas  <<<Ilustración 7.10>>>
# Se emplea parte del código propuesto en repositorio Stack Overflow
#https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot

#Definición de función empleada más adelante para generar el gráfico
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    #plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("Componente Principal PC{}".format(1))
    plt.ylabel("Componente Principal PC{}".format(2))
    plt.title("Gráfico de cargas/importancias de las variables en PC1 y PC2 \n ")
    plt.grid()
    plt.show()

pca.fit_transform(x)[0:809]
myplot(pca.fit_transform(x)[0:809],np.transpose(pca.components_[0:809, :]),df5x.columns)
plt.show()

# Gráfico de proporción de varianza explicada por ponente principal <<<Ilutración 7.12>>>
#Se emplea código original de Lorraine Li disponible en https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad
matriz_cov = np.cov(X_pca.T)
autovalores, autovectores = np.linalg.eig(matriz_cov)
tot = sum(autovalores)
var_exp = [(i / tot) for i in sorted(autovalores, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.7,
        align='center', label='Varianza explicada por PC')
plt.step(range(1,14), cum_var_exp, where='mid',
         label='Varianza explicada acumulada (escalonada)')
plt.plot(range(1,14), cum_var_exp,
         label='Varianza explicada acumulada (con pendiente)', linewidth=0.5, color='r')
plt.ylabel('Proporción de varia explicada')
plt.xlabel('Número de componentes principales empleadas')
plt.title('Proporción de varianza explicada por PC')
plt.legend()
plt.grid()
plt.show()



