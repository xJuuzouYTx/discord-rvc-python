import os
import zipfile

# Obtén el directorio actual
directorio_actual = os.getcwd()

# Lista todos los archivos en el directorio actual
archivos_en_directorio = os.listdir(directorio_actual)

# Itera a través de los archivos para buscar archivos ZIP
for archivo in archivos_en_directorio:
    if archivo.endswith(".zip"):
        # Obtiene el nombre del archivo ZIP sin la extensión
        nombre_sin_extension = os.path.splitext(archivo)[0]

        # Crea una carpeta con el mismo nombre que el archivo ZIP
        carpeta_destino = os.path.join(directorio_actual, nombre_sin_extension)

        # Crea la carpeta si no existe
        if not os.path.exists(carpeta_destino):
            os.mkdir(carpeta_destino)

        # Abre el archivo ZIP
        with zipfile.ZipFile(archivo, 'r') as zip_ref:
            # Extrae todo el contenido en la carpeta de destino
            zip_ref.extractall(carpeta_destino)

        # Elimina el archivo ZIP después de la extracción
        os.remove(archivo)

        print(f"Se extrajo el contenido de '{archivo}' en '{carpeta_destino}' y se eliminó el archivo ZIP.")

