import sys
import pandas
import pandas as pd
import csv
import time
import datetime
import numpy as np
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)

#PARTE 1: Conversión formato

 #Bucle de transformación de ficheros deformato JSON a CSV

print('Inicio del proceso:',time.asctime( time.localtime(time.time())))
meses_registrados= ('201704','201705','201706','201707','201708','201709','201710','201711','201712','201801','201802','201803','201804','201805','201806','201807','201808','201809','201810','201811','201812','201901','201902','201903','201904','201905','201906')
path_origen= 'C:\\MYPATH\\TFG\\Datos\\JSON\\'
path_destino = 'C:\\MYPATH\\TFG\\Datos\\DataSet\\'
datosGlobal=pd.DataFrame()

for mensualidad in meses_registrados:
    #Lectura de ficheros JSON haciendo uso de 'chunksize'
    chunks = []
    for chunk in pd.read_json(path_origen + mensualidad +'_Usage_Bicimad.json',chunksize=20000,encoding="latin-1", lines=True, orient='values',dtype={'_id':str,'ageRange':int, 'idplug_base':str, 'idplug_station' : str, 'idunplug_base': str, 'idunplug_station': str, 'travel_time': int, 'unplug_hourTime':str, 'user_day_code': str, 'user_type':str , 'zip_code':str}):
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)

    #Condición de eliminación del campo TRACKS, si existe
    if df.columns.isin(['track']).any():
        df = df.drop('track', 1) #valor 1 indica que actúa sobre las columnas, 0 para elimitar filas
        print('El campo TRACK ha sido suprimido de',mensualidad+'_Usage_Bicimad')



#PARTE 2: filtrado
#Filtro campo '_id'
df['_id'] = df['_id'].str.replace('$', "",regex=True)
df['_id'] = df['_id'].str.replace("{'oid': '", "",regex=True)
df['_id'] = df['_id'].str.replace("'}", "",regex=True)
#Filtro campo 'unplug_hourTime'
df['unplug_hourTime'] = df['unplug_hourTime'].str.replace('$', "",regex=True)
df['unplug_hourTime'] = df['unplug_hourTime'].str.replace("{'date': '", "",regex=True)
df['unplug_hourTime'] = df['unplug_hourTime'].str.replace("'}", "",regex=True)

#Supresión de registros duplicados
df.drop_duplicates(keep = 'first', inplace = True)
 


#PARTE 3: Creación de variables temporales y clave primaria

#Campo 'unplug_hourTime' convertido en formato Datetime
df['unplug_hourTime']=pd.to_datetime(df['unplug_hourTime'], utc=False)
#TIEMPOS
df=pd.DataFrame()
df=df
df['year'] = df['unplug_hourTime'].dt.year
df['month'] = df['unplug_hourTime'].dt.month
df['day'] = df['unplug_hourTime'].dt.day

#Fecha como clave primaria con meteo
df['datetime']=df['unplug_hourTime'].dt.year.astype(str)+'/'+df['unplug_hourTime'].dt.month.astype(str)+'/'+df['unplug_hourTime'].dt.day.astype(str)

#Visualización del resultado
df.head()
df.tail()



#PARTE 4: Almacenamiento local
#Copia de datos en nuevo DataFrame (en caso de recuperación inmediata, ahorra el tiempo de lectura)
df1=pd.DataFrame()
df1=df

#Generación de fichero CSV
df.to_csv(path_destino+'parcial_trayectosBiciMAD_v1.csv', encoding='utf-8', index=False)






