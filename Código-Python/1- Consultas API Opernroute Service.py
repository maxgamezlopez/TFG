import sys
import pandas
import pandas as pd
import csv
import time
import requests

#Importación de datos de trayectos únicos registrados
print('Inicio del proceso:',time.asctime( time.localtime(time.time())))
trayectos = pd.read_csv("C:\\MYPATH\\TFG\\API\\trayectos\\trayectosunicos_coordenadas.csv", delimiter=';')
longitud=len(trayectos)
SalidaAPI = pd.DataFrame(columns=['id', 'origen', 'final','status','reason','text'])
path_destino='C:\\MYPATH\\TFG\\API\\trayectos\\output\\'


#API restringe 2000 consultas/key-día. Se preparan varios ciclos con varios accesos.
inicio_hoy=0  #A configurar en cada grupo de ciclos
fin_hoy=3000  #A configurar en cada grupo de ciclos

#CICLO 1
inicio=0 #A configurar en cada ciclo
fin=1500 #A configurar en cada ciclo
APIkey='xxxxxxxxxx1110001cf6248a3a53c0b7aaa4baebd56b3xxxxxxxxxx' #Código de acceso (API key). Contenido omitido, puede puede obtener una en https://openrouteservice.org/

# Bucle donde se suceden todas las consultas lanzadas en cada ciclo
for i in range(inicio,(fin),1): 
    #Cuerpo de la consulta
    body = {"coordinates":[[trayectos.loc[i][4],trayectos.loc[i][3]],[trayectos.loc[i][6],trayectos.loc[i][5]]] }
    headers = {'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8','Authorization': APIkey}
    call = requests.post('https://api.openrouteservice.org/v2/directions/cycling-electric', json=body, headers=headers) #Dirección URL de la consulta. Se especifica modalidad de transporte.

    print(call.status_code, call.reason)
    print(call.text)
    
    SalidaQuery=pd.DataFrame([[trayectos.loc[i][0],trayectos.loc[i][1],trayectos.loc[i][2],call.status_code,call.reason,call.text]],columns=['id', 'origen', 'final','status','reason','text'])
    SalidaAPI=SalidaAPI.append(SalidaQuery) #Resultado incluido en el DataFrame Global
    time.sleep(1.4) #Tiempo de espera necesario para no exceder 40 consultas por minuto (restricción de aplicación)
    print('Completadas',i,'de',len(trayectos),'consultas.') #Salida del progreso del ciclo

SalidaAPI.to_csv(path_destino + 'salidaAPI('+str(inicio_hoy)+'-'+str(fin_hoy)+')resto.csv', encoding='utf-8', index=False, sep=';') #Almacenamiento local en CSV



#CICLO 2
inicio=1500
fin=3000
APIkey='xxxxxxxxxx243501deh648aa2001ba4h13245622baddaxxxxxxxxxx' #Código de acceso (API key). Contenido omitido, puede puede obtener una en https://openrouteservice.org/
for i in range(inicio,(fin),1): 
    body = {"coordinates":[[trayectos.loc[i][4],trayectos.loc[i][3]],[trayectos.loc[i][6],trayectos.loc[i][5]]] }
    headers = {'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8','Authorization': APIkey}
    call = requests.post('https://api.openrouteservice.org/v2/directions/cycling-electric', json=body, headers=headers)

    print(call.status_code, call.reason)
    print(call.text)
    
    SalidaQuery=pd.DataFrame([[trayectos.loc[i][0],trayectos.loc[i][1],trayectos.loc[i][2],call.status_code,call.reason,call.text]],columns=['id', 'origen', 'final','status','reason','text'])
    SalidaAPI=SalidaAPI.append(SalidaQuery)
    time.sleep(1.4)
    print('Completadas',i,'de',len(trayectos),'consultas.')

SalidaAPI.to_csv(path_destino + 'salidaAPI('+str(inicio_hoy)+'-'+str(fin_hoy)+')resto.csv', encoding='utf-8', index=False, sep=';') #Almacenamiento local en CSV. Sobrescribe al anterior.

#Tantos ciclos por día como códigos de acceso, o códigos pendientes de consultar, se dispongan.

print('Fin del proceso:',time.asctime( time.localtime(time.time())))
