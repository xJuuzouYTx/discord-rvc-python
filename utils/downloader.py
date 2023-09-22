import os
import shutil
from mega import Mega
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import gdown
import re
import wget
import sys

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
gdrive = GoogleDrive(gauth)

url_modelo = "https://drive.google.com/file/d/12ltzXpNSpHmJtyH1VU1XzmvzvktgikKU/view"

# define Python user-defined exceptions
class InvalidDriveId(Exception):
  def __init__(self, message="Error de la url"):
        self.message = message
        super().__init__(self.message)

def model_downloader(url, zip_path, dest_path):
  def drive_download(url, dest_folder):
      # Descargar el archivo de Google Drive usando gdown
      try:
          print(f"Descargando desde drive...")

          # Obtener id del archivo y descargar
          id_archivo = url.split('/')[-1]
          match = re.match(r"https://drive\.google\.com/.*[?&]id=([^/&]+)", url)
          if match:
              id = match.group(1)
          else:
              match = re.match(r"https://drive\.google\.com/file/d/([^/&]+)", url)
              if match:
                  id = match.group(1)
              else:
                  id = None
          id_archivo = id

          if not id_archivo:
            raise InvalidDriveId("No se encontró el ID del archivo, compruebe la url y vuelva a intentarlo.")

          downloaded = gdrive.CreateFile({'id':id_archivo})
          dest_path = os.path.join(dest_folder, downloaded['title'])
          downloaded.GetContentFile(dest_path)
          return downloaded['title']

      except InvalidDriveId as idErr:
        print(idErr)
        return None
      except Exception as e:
          print("Ocurrio un error en drive, reintentando")
          try:
            filename = gdown.download(url, fuzzy=True)
            shutil.move(os.path.join(os.getcwd(), filename), os.path.join(dest_folder, filename))
            return filename
          except:
            print("El intento de descargar con drive no funcionó")
            return None

  def mega_download(url, dest_folder):
    try:
      file_id = None
      if "#!" in url:
        file_id = url.split("#!")[1].split("!")[0]
      elif "file/" in url:
        file_id = url.split("file/")[1].split("/")[0]
      else:
        file_id = None

      print(f"Descargando desde mega...")
      if file_id:
        mega = Mega()
        m = mega.login()
        filename = m.download_url(url,dest_path=dest_folder)
        
        return os.path.basename(filename)
      else:
        return None

    except Exception as e:
      print("Ocurrio un error**")
      print(e)
      return None

  def download(url, dest_folder):
    try:
      print(f"Descargando desde url generica...")
      dest_path = wget.download(url=url, out=dest_folder)
      
      return os.path.basename(dest_path)
    except Exception as e:
      print(f"Error al descargar el archivo: {str(e)}")

  filename = ""
  
  if not os.path.exists(zip_path):
    os.mkdir(zip_path)

  if url and 'drive.google.com' in url:
      # Descargar el elemento si la URL es de Google Drive
    filename = drive_download(url, zip_path)
  elif url and 'mega.nz' in url:
    filename = mega_download(url, zip_path)
  elif url and 'pixeldrain' in url:
    print("No se puede descargar de pixeldrain")
    sys.exit()
  else:
    filename = download(url, zip_path)

  if filename:
    print(f"Descomprimiendo {filename}...")
    modelname = str(filename).replace(".zip","")
    shutil.unpack_archive(os.path.join(zip_path, filename), os.path.join(dest_path, modelname))

    return modelname
  else:
    return None