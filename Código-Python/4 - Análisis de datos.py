import sys
import pandas
import pandas as pd
import csv
import time
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)


#Lectura de CSV donde se alamacenan los registros completos
chunks = []
 for chunk in pd.read_csv('D:\\CS PC MAXI\\Documentos\\ETSII\\TFG\\Datos\\DataSet\\completo_desplazamientoORSmeteo_v1.csv', sep=',', chunksize=2000000, low_memory=False):
    chunks.append(chunk)
df = pd.concat(chunks, axis=0)
#df['unplug_hourTime']=pd.to_datetime(df['unplug_hourTime'], utc=False)

#Pasa emplearse df2 como copia de df. Con ello se ahorra tiempo en caso de querer recuperar la información original al no tener que leer de nuevo el CSV
df2=df

#Exploración de número de registros nulos por columna contemplada en el conjunto de datos.
for i in df2.columns:
    print(i+':'+ str(len(df2[df2[i].isnull()])))

#Se define df3 como muestra aleatoria de 100.000 registros de df2 (sampleado)
df3=df2.sample(n=100000,random_state=1)  #random_state como semilla de generador de números aleatorios


#Descripción de variables <<< Ilustración 6.1 >>>
df2.describe()

#Diagrama de SAMPLEADO Age_range <<< Ilustración 6.2>>>
x_labels = ['No identificado','< 17 años','17-18 años','19-26 años','27-40 años','41-65 años','>65 años']
df3=df2['ageRange'].sample(n=100000,random_state=1).value_counts().sort_index().head(10)
df2['ageRange'].sample(n=100000,random_state=1).value_counts().sort_index().plot(kind='bar',color='#002e5d').set_xticklabels(x_labels)
plt.title('Número de desplazamientos por rango de edad \n muestra n=100.000')
plt.ylabel('Número de desplazamientos')
plt.show()

#Diagrama de bigotes sobre muestra de 100.000 registros (con outliers)  <<< Ilustraciones 6.3 y 6.4 >>>
df2.sample(n=100000,random_state=1).boxplot(column=['travel_time', 'duration'])
plt.title('Diagrama de caja de variables travel_time y duration \n muestra n=100.000')
plt.xlabel('')
plt.ylabel('Segundos')
plt.show()

#Tiempo real frente predicho por Openroute Service   <<< Ilustración 6.5>>>
df2.sample(n=10000,random_state=1).plot.scatter(y='travel_time', x=['duration'], figsize=(10,10), grid=True,s=0.5)
x = [1,3000]
y = [1,3000]
plt.plot(x,y, color="red", label = 'línea de referencia: tiempo previsto igual al de desplazamiento', linewidth=1 )
plt.title('Duración prevista por Openroute Service frente a tiempo empleado \n muestra n=10.000')
plt.xlabel('Duración prevista')
plt.ylabel('Tiempo empleado')
plt.show()

#Histograma de muestreo de variable travel_time.  <<< Ilustración 6.6  >>>
df2['travel_time'].sample(n=100000,random_state=1).plot.hist(bins=50000, xlim=(0,2000),color='#002e5d')
plt.title('Histograma de tiempos de viaje \n muestra n=100.000')
plt.xlabel('Segundos')
plt.ylabel('Frecuencia')
plt.show()

#Histograma de registros menores a 60s en variable travel_time registros .  <<< Ilustración 6.7  >>>
df4=df2[df2.travel_time < 60]
df4.info()
df4['travel_time'].plot.hist(bins=4000, xlim=(0,60),color='#002e5d')
plt.title('Histograma de tiempos de viaje < 60 segundos')
plt.xlabel('Segundos')
plt.ylabel('Frecuencia')
plt.show()


#Diagrama frecuencia por meses (muestreo) <<<Ilustración 6.8>>>
#Se eliminan anteriores al 1 de julio de 2017 (contempla julio 2017 - junio 2019  24 meses)
df5=df2.drop(df2[df2.datetime <2017063100].index)

x_labels = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
df5['month'].sample(n=100000,random_state=1).value_counts().sort_index().plot(kind='bar',color='#002e5d').set_xticklabels(x_labels)
plt.title('Número de desplazamientos por meses \n muestra n=100.000 \n (julio 2017 - junio 2019)')
plt.ylabel('Número de desplazamientos')
plt.show()

#Diagrama frecuencia por días de la semana (muestreo) <<<Ilustración 6.9>>>
#Se eliminan anteriores al 1 de julio de 2017 (contempla julio 2017 - junio 2019  24 meses)
#Diagrama de SAMPLEADO weekday
df3=df2.drop(df2[df2.datetime <2017040300].index)
x_labels = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
df4=df3['weekday'].sample(n=100000,random_state=1).value_counts().sort_index()
df5=pd.DataFrame({df4['Monday'], df4['Tuesday'],df4['Wednesday'],df4['Thursday'],df4['Friday'],df4['Saturday'],df4['Sunday']})
df5.sort_index().plot(kind='bar',color='#002e5d',legend=False).set_xticklabels(x_labels)
plt.title('Número de desplazamientos por día de la semana \n muestra n=100.000')
plt.ylabel('Número de desplazamientos')
plt.show()

#Temperaturas máximas y mínimas por días <<<Ilustración 6.10>>>
df2['bydate']=df2['year'].astype(str)+'-'+df2['month'].astype(str)+'-'+df2['day'].astype(str)
s1=df2.groupby(['bydate'])['meteo_tmax'].agg(lambda x: x.unique().mean()).plot(linewidth=0.5,color='#ff0000')
s2=df2.groupby(['bydate'])['meteo_tmin'].agg(lambda x: x.unique().mean()).plot(linewidth=0.5,color='#002e5d')
plt.title('Temperaturas máximas y mínimas por día')
plt.ylabel('Temperatura (ºC)')
plt.xlabel('Fecha')
plt.legend()
plt.show()


#Número de desplazamientos por día y hora <<<Ilustración 6.11 (superior)>>>
df2['bydate']=df2['unplug_hourTime'].astype(str).str[0:13]
df2['bydate'].value_counts().sort_index().plot(linewidth=0.5,color='#002e5d')
plt.title('Desplazamientos por hora')
plt.ylabel('Número de desplazamientos')
plt.show()

#Número de desplazamientos por días <<<Ilustración 6.11 (inferior)>>>
df2['bydate']=df2['unplug_hourTime'].astype(str).str[0:10]
df2['bydate'].value_counts().sort_index().plot(linewidth=0.5,color='#002e5d')
plt.title('Desplazamientos por día')
plt.ylabel('Número de desplazamientos')
plt.show()


#Duración mediana por días <<<Ilustración 6.12>>>
df2['bydate']=df2['unplug_hourTime'].astype(str).str[0:10]
df3=df2.groupby('bydate')['travel_time'].median()
df3[(df3 < 10000) & (df3 > 300)].plot(linewidth=0.5,color='#002e5d')
plt.title('Duración mediana por día')
plt.xlabel('Fecha')
plt.show()



#Flujo neto (enganches-desenganches) en la estación 1 (Puerta del Sol)
id_estacion=1
flujo=df2[np.logical_or(df2['idunplug_station'].astype(int)==id_estacion , df2['idplug_station'].astype(int)==id_estacion)]
flujo['diaHora']=flujo['bydate']+'T'+flujo['datetime'].astype(str).str[8:10]
flujo_entrada=flujo[flujo['idplug_station'].astype(int)==id_estacion]['diaHora'].value_counts()
flujo_salida=flujo[flujo['idunplug_station'].astype(int)==id_estacion]['diaHora'].value_counts()
flujo_entrada=pd.DataFrame({'diaHora':flujo_entrada.index, 'entradas':flujo_entrada.values})
flujo_salida=pd.DataFrame({'diaHora':flujo_salida.index, 'salidas':flujo_salida.values})
flujo=flujo_entrada.merge(flujo_salida, on='diaHora')
flujo['neto']=flujo['entradas']-flujo['salidas']
flujo.groupby(['diaHora'])['neto'].agg(lambda x: x.unique().mean()).plot(linewidth=0.5,color='#002e5d')
plt.title('Flujo neto de estación 1 (Puerta del Sol) \n (diferencia entre enganches y desenganches para cada hora)')
plt.ylabel('Flujo neto')
plt.xlabel('Fecha y hora')
plt.legend('')
plt.show()









df2['unplug_hourTime'].value_counts().sort_index().plot(linewidth=0.5,color='#002e5d')
plt.xaxis.set_major_locator(mdates.YearLocator())
plt.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
plt.title('Desplazamientos por hora y día')
plt.ylabel('Número de desplazamientos iniciados')
plt.show()

#Estudio por días
df2['bydate']=df2['unplug_hourTime'].astype(str).str[0:10]
df2['bydate'].value_counts().sort_index().
plt.title('Desplazamientos por día')
plt.ylabel('Número de desplazamientos')
plt.show()



df3[df3 < 300]
df2.head()



#Estudio por días-hora
df2['bydate']=df2['unplug_hourTime'].astype(str).str[0:13]
df2['bydate'].value_counts().sort_index().plot(linewidth=0.5,color='#002e5d')
plt.title('Desplazamientos por hora')
plt.ylabel('Número de desplazamientos')
plt.show()

#Temperatura máxima  por día
df2.groupby(['bydate'])['meteo_tmax'].agg(lambda x: x.unique().mean()).plot(linewidth=0.5,color='#002e5d')
flujo=df2[np.logical_or(df2['idunplug_station'].astype(int)==1 , df2['idplug_station'].astype(int)==1)]
flujo.head()
df2.columns
fecha=df2['bydate'].value_counts()
fecha.head()



#DESCRIPCIÓN ESTACIÓN 1 Y DATASET

chunks = []
 for chunk in pd.read_csv('D:\\CS PC MAXI\\Documentos\\ETSII\\TFG\\Datos\\DataSet\\flujoneto_meteo_est1.csv', chunksize=2000000, low_memory=False, sep=';'):
    chunks.append(chunk)
tsu = pd.concat(chunks, axis=0)

tsu.head()
tsu=tsu.sort_values(by=['diahora'])

s1=tsu.groupby(['bydate'])['meteo_tmax'].agg(lambda x: x.unique().mean()).plot(linewidth=0.5,color='#ff0000')
    
plt.plot( 'x', 'y1', data=s1, marker='o', markerfacecolor='#002e5d', markersize=12, color='#002e5d', linewidth=4)
plt.plot( 'x', 'y2', data=s2, marker='x', markerfacecolor='#ff0000'color='#ff0000', linewidth=4)
plt.title('Temperaturas máximas y mínimas por día')
plt.ylabel('Temperatura (ºC)')
plt.xlabel('Fecha')
plt.legend()
plt.show()

tsu.head()
tsu['cero']=0
tsu=tsu.sort_values(['diahora'], ascending=[True])
len(tsu)

tsu.plot(x='diahora', y=['flujo','cero'], figsize=(10,10), grid=False,linewidth=0.5)
plt.title('Flujo neto de estación 1 (Puerta del Sol) \n (diferencia entre enganches y desenganches para cada hora)')
plt.ylabel('Flujo neto')
plt.xlabel('Fecha y hora')
plt.legend('')
plt.show()


#Tiempo real frente predicho por Openroute Service
    df2=df
    df2.sample(n=10000,random_state=1).plot.scatter(y='travel_time', x=['duration'], figsize=(10,10), grid=True,s=0.5)
    x = [1,3000]
    y = [1,3000]
    plt.plot(x,y, color="red", label = 'línea de referencia: tiempo previsto igual al de desplazamiento', linewidth=1 )
    plt.title('Duración prevista por Openroute Service frente a tiempo empleado \n muestra n=10.000')
    plt.xlabel('Duración prevista')
    plt.ylabel('Tiempo empleado')
    plt.show()

df2.head()

