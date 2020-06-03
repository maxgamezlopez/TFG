#Instrucciones

Se puede disponer de los modelos descargando los archivos correspondientes y cargándolos con Python

  modelo = joblib.load('C:\\MYPATH\\Modelo v1')

Donde MYPATH\\Modelo v1 representan la localización y título del archivo.

Para obterner nueva predicción:
   modelo.predict(X_nuevo)
   
Donde X_nuevo representa los registros de entrada.



