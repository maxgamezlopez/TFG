import sys
import pandas
import pandas as pd
import csv
import time
import datetime
pd.set_option('display.max_columns', 50)


#Importación datos de desplazamientos de BiciMAD.
chunks = []
 for chunk in pd.read_csv('C:\\MYPATH\\TFG\\API\\trayectos\\output\\parcial_trayectosBiciMAD_v1.csv', chunksize=2000000, low_memory=False,dtype={'_id':str,'year':int, 'month':int, 'day':int, 'datetime':str, ageRange':int, 'idplug_base':str, 'idplug_station' : str, 'idunplug_base': str, 'idunplug_station': str, 'travel_time': int, 'unplug_hourTime': str, 'user_day_code': str, 'user_type':str , 'zip_code':str}):
    chunks.append(chunk)
df = pd.concat(chunks, axis=0)

df.head() #Visualización


#Importación datos consulta API Openroute Service
#Requiere de procesamiento previo: unificación de ficheros de respuestas de consulta, separación en columnas y reemplazo (compatible con Excel)
#Nota: se recomienda previa aplicación de la subsanación de omitidos (p.e.: imputación de valores omitidos)
ORSquery=pd.read_csv('C:\\MYPATH\\TFG\\API\\trayectos\\output\\parcial_ORS_globalv1.csv', sep=';', low_memory=False,dtype={'id':str,'keyTry':str, 'origen':str, 'destino' : str, 'reason': str, 'coords': str, 'distance': str, 'duration': str})
ORSquery.head() #Visualización
ORSmerge=ORSquery.drop(['id','origen','destino','status','reason','coords'],axis=1)
ORSmerge.head() #Visualización


#Creación de nuevo campo que sirve como clave primaria entre desplazamientos y ORS:
df['keyTry']=df['idunplug_base']+'-'+df['idplug_base']
df['keyTry'].head() #Visualización
df=df.sort_values('unplug_hourTime') #Registros ordenados cronológicamente

#Asociación entre desplazamientos y respuesta de ORS
df2=pd.merge(df,ORSmerge) 
df2.head() #Visualización



#Importación registros meteorológicos de AEMET.
#Consulta a través de https://opendata.aemet.es/, almacenamiento y preprocesamiento posterior (compatible con excel).

chunks = []
 for chunk in pd.read_csv('D:\\CS PC MAXI\\Documentos\\ETSII\\TFG\\API\\AEMET\\reportesAEMET(reducido).csv', chunksize=2000000, low_memory=False,sep=';'):
    chunks.append(chunk)
meteo = pd.concat(chunks, axis=0)

#Cambio de separador decimal de comas a puntos y supresión código en conjunto (compatible variable tipo float)
meteo['meteo_tmed'] = meteo['meteo_tmed'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_prec'] = meteo['meteo_prec'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_tmin'] = meteo['meteo_tmin'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_tmax'] = meteo['meteo_tmax'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_dir'] = meteo['meteo_dir'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_velmedia'] = meteo['meteo_velmedia'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_racha'] = meteo['meteo_racha'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_presMax'] = meteo['meteo_presMax'].astype(str).str.replace(',', ".",regex=True)
meteo['meteo_presMin'] = meteo['meteo_presMin'].astype(str).str.replace(',', ".",regex=True)

#Nota: la creación de un registro de año / mes / día en el conjunto de meteo debe ir en consonancia con el formato de fecha considerado en la clave primaria de los desplazamientos.
#Por ejemplo:  2017/04/01 mantine formato diferente a 2017/4/1.

df=pd.merge(df,meteo) #Asociación entre desplazamientos y registros meteo de AEMET

df=df.drop_duplicates() #Supresión de registros duplicados

df.to_csv('D:\\CS PC MAXI\\Documentos\\ETSII\\TFG\\Datos\\DataSet\\DatosGlobaltrymeteo.csv', encoding='utf-8', index=False, sep=';')





