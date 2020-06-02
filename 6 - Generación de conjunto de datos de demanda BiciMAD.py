import sys
import pandas
import pandas as pd
import csv
import time
import datetime
import numpy as np

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 850)


#Importación de conjuntos de datos de desplazamientos
chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\Datos\\DataSet\\parcial_trayectosBiciMAD_v2.csv', sep=';', chunksize=2000000, low_memory=False):
    chunks.append(chunk)
df = pd.concat(chunks, axis=0)


#Conteo de número de desplazamientos por días y exportación del número en un nuevo DataFrame
df3=df.groupby([df['year'], df['month'], df['day']]).count()
df4=pd.DataFrame()
df4[['_id']]=df3[['_id']]

#Creación de nuevas columnas. 'Count' corresponde a secuencia cronológica que permita identificar posibles tendencias a lo largo del tiempo.
df4['index1'] = df4.index
df4=df4.sort_index(by=['year','month','day'])
df4['meteo_date']=""
df4['meteo_date']=df4['meteo_date'].astype(str)
df4['datetime']=""
df4['datetime']=df4['datetime'].astype(str)
df4['year']=""
df4['month']=""
df4['day']=""
df4['weekday_int']=""
df4['weekday_str']=""
df4['count']=""


#Generacíón de datetime y meteo_date (clave primaria) a partir de index1 (tupla)
for i in range(0,len(df4)-1,1):
    j=df4['index1'][i]
    df4['count'][i]=i
    if (len(str(j[1])) == 1):
        k='0'+str(j[1])
    else:
        k=str(j[1])     
    if (len(str(j[2])) == 1):
        l='0'+str(j[2])
    else:
        l=str(j[2]) 
    df4['datetime'][i]=str(j[0])+k+l
    df4['meteo_date'][i]=(str(l)+'/'+str(k)+'/'+str(j[0]))
    df4['year'][i]=j[0]
    df4['month'][i]=j[1]
    df4['day'][i]=j[2]
    df4['weekday_int'][i]=datetime.datetime(j[0],j[1], j[2]).weekday()+1
    df4['weekday_str'][i]=datetime.datetime(j[0],j[1], j[2]).strftime('%A')

#Importación de registros meteorológicos (sin valores nulos)
chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\API\\AEMET\\reportesAEMET(sin missingdata).csv', chunksize=200, low_memory=False,sep=','):
    chunks.append(chunk)
meteo = pd.concat(chunks, axis=0)

#Asociación a registros meteorológicos con 'meteo_date' como clave primaria.
df5=pd.merge(df4,meteo.drop(['year','month','day'],axis=1))

#Almecenamiento local (en formato CSV) de conjunto de datos correpsondiente a la demanda del sistema BiciMAD
df5.to_csv('C:\\MYPATH\\TFG\\Datos\\DataSet\\demandaBiciMAD_v1.csv', encoding='utf-8', index=False)


