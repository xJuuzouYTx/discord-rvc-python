import requests
import os
import ffmpeg
import numpy as np
import random
from pathlib import Path
import subprocess
import shutil
import tarfile

class Audio:
    
    audio_path = "./audios"
    
    def __init__(self, name, url):
        self._name = name
        self._url = url
          
        if not os.path.exists(Audio.audio_path):
            os.mkdir(Audio.audio_path)
        
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
        
    @property
    def url(self):
        return self._url
    
    @url.setter
    def url(self, url):
        self._url = url
        
    def __str__(self):
        return f'Audio: {self._name} {self._url}'
    
    @classmethod
    def load_audio(self, file, sr, do_formant, Quefrency, Timbre, stft="stftpitchshift.exe"):
        """
        Carga y procesa un archivo de audio.

        Args:
            file (str): El nombre del archivo de audio.
            sr (int): Tasa de muestreo deseada.
            do_formant (bool): Indica si se debe aplicar el procesamiento de formantes.
            Quefrency (float): Valor de quefrencia para el procesamiento de formantes.
            Timbre (float): Valor de timbre para el procesamiento de formantes.
            stft (stft): Especifica la herramienta de procesamiento de audio.

        Returns:
            np.ndarray: Un array NumPy con los datos de audio procesados.
        """
        
        converted = False
        try:
            # Eliminar espacios y comillas del nombre del archivo
            file = (file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")) 
            file_formanted = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            
            if (do_formant):
                numerator = round(random.uniform(1, 4), 4)

                if not file.endswith(".wav"):
                    if not os.path.isfile(f"{file_formanted}.wav"):
                        converted = True
                        
                        converting = (
                            ffmpeg.input(file_formanted, threads=0)
                            .output(f"{file_formanted}.wav")
                            .run(
                                cmd=["ffmpeg", "-nostdin"],
                                capture_stdout=True,
                                capture_stderr=True,
                            )
                        )
                    else:
                        pass

                file_formanted = (
                    f"{file_formanted}.wav"
                    if not file_formanted.endswith(".wav")
                    else file_formanted
                )

                # Aplicar procesamiento de formantes
                os.system(
                    '%s -i "%s" -q "%s" -t "%s" -o "%sFORMANTED_%s.wav"'
                    % (
                        stft,
                        file_formanted,
                        Quefrency,
                        Timbre,
                        file_formanted,
                        str(numerator),
                    )
                )

                print(f" · Formanted {file_formanted}!\n")

                out, _ = (
                    ffmpeg.input(
                        "%sFORMANTED_%s.wav" % (file_formanted, str(numerator)), threads=0
                    )
                    .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                    .run(
                        cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                    )
                )

                try:
                    os.remove("%sFORMANTED_%s.wav" % (file_formanted, str(numerator)))
                except Exception:
                    pass
                    print("couldn't remove formanted type of file")

            else:
                out, _ = (
                    ffmpeg.input(file, threads=0)
                    .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                    .run(
                        cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

        if converted:
            try:
                os.remove(file_formanted)
            except Exception:
                pass
                print("couldn't remove converted type of file")
            converted = False

        return np.frombuffer(out, np.float32).flatten()
    
    @classmethod
    def dowload_from_url(self, url = None, output = "./audios/file.wav"):
        """
        Descarga un aduio desde una url
        Args:
            path: Folder where the audio will be downloaded
        Returns:
            return: the path of the downloaded audio
        """
        request = requests.get(url, allow_redirects=True)
        open(output, 'wb').write(request.content)
        
        return output
    
    
def install_dependencies(ForceUpdateDependencies, ForceTemporaryStorage):
    # Mounting Google Drive
    if not ForceTemporaryStorage:
        from google.colab import drive

        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
        else:
            print('Drive is already mounted. Proceeding...')

    # Function to install dependencies with progress
    def install_packages():
        packages = ['build-essential', 'python3-dev', 'ffmpeg', 'aria2']
        pip_packages = ['pip', 'setuptools', 'wheel', 'httpx==0.23.0', 'faiss-gpu', 'fairseq', 'gradio==3.34.0',
                        'ffmpeg', 'ffmpeg-python', 'praat-parselmouth', 'pyworld', 'numpy==1.23.5',
                        'numba==0.56.4', 'librosa==0.9.2', 'mega.py', 'gdown', 'onnxruntime', 'pyngrok==4.1.12',
                        'gTTS', 'elevenlabs', 'wget', 'tensorboardX', 'unidecode', 'huggingface-hub', 'stftpitchshift==1.5.1', 
                        'yt-dlp', 'pedalboard', 'pathvalidate', 'nltk', 'edge-tts', 'git+https://github.com/suno-ai/bark.git', 'python-dotenv' , 'av']

        print("Updating and installing system packages...")
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call(['apt-get', 'install', '-qq', '-y', package])

        print("Updating and installing pip packages...")
        subprocess.check_call(['pip', 'install', '--upgrade'] + pip_packages)


        print('Packages up to date.')

    # Function to scan a directory and writes filenames and timestamps
    def scan_and_write(base_path, output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for dirpath, dirs, files in os.walk(base_path):
                for filename in files:
                    fname = os.path.join(dirpath, filename)
                    try:
                        mtime = os.path.getmtime(fname)
                        writer.writerow([fname, mtime])
                    except Exception as e:
                        print(f'Skipping irrelevant nonexistent file {fname}: {str(e)}')
        print(f'Finished recording filesystem timestamps to {output_file}.')

    # Function to compare files
    def compare_files(old_file, new_file):
        old_files = {}
        new_files = {}

        with open(old_file, 'r') as f:
            reader = csv.reader(f)
            old_files = {rows[0]:rows[1] for rows in reader}

        with open(new_file, 'r') as f:
            reader = csv.reader(f)
            new_files = {rows[0]:rows[1] for rows in reader}

        removed_files = old_files.keys() - new_files.keys()
        added_files = new_files.keys() - old_files.keys()
        unchanged_files = old_files.keys() & new_files.keys()

        changed_files = {f for f in unchanged_files if old_files[f] != new_files[f]}

        for file in removed_files:
            print(f'File has been removed: {file}')

        for file in changed_files:
            print(f'File has been updated: {file}')

        return list(added_files) + list(changed_files)

    # Check if CachedRVC.tar.gz exists
    if ForceTemporaryStorage:
        file_path = '/content/CachedRVC.tar.gz'
    else:
        file_path = '/content/drive/MyDrive/RVC_Cached/CachedRVC.tar.gz'

    content_file_path = '/content/CachedRVC.tar.gz'
    extract_path = '/'

    if not os.path.exists(file_path):
        folder_path = os.path.dirname(file_path)
        os.makedirs(folder_path, exist_ok=True)
        print('No cached dependency install found. Attempting to download GitHub backup..')

        try:
            download_url = "https://github.com/kalomaze/QuickMangioFixes/releases/download/release3/CachedRVC.tar.gz"
            subprocess.run(["wget", "-O", file_path, download_url])
            print('Download completed successfully!')
        except Exception as e:
            print('Download failed:', str(e))

            # Delete the failed download file
            if os.path.exists(file_path):
                os.remove(file_path)
            print('Failed download file deleted. Continuing manual backup..')

    if Path(file_path).exists():
        if ForceTemporaryStorage:
            print('Finished downloading CachedRVC.tar.gz.')
        else:
            print('CachedRVC.tar.gz found on Google Drive. Proceeding to copy and extract...')

        # Check if ForceTemporaryStorage is True and skip copying if it is
        if ForceTemporaryStorage:
            pass
        else:
            shutil.copy(file_path, content_file_path)

        print('Beginning backup copy operation...')

        with tarfile.open(content_file_path, 'r:gz') as tar:
            for member in tar.getmembers():
                target_path = os.path.join(extract_path, member.name)
                try:
                    tar.extract(member, extract_path)
                except Exception as e:
                    print('Failed to extract a file (this isn\'t normal)... forcing an update to compensate')
                    ForceUpdateDependencies = True
            print(f'Extraction of {content_file_path} to {extract_path} completed.')

        if ForceUpdateDependencies:
            install_packages()
            ForceUpdateDependencies = False
    else:
        print('CachedRVC.tar.gz not found. Proceeding to create an index of all current files...')
        scan_and_write('/usr/', '/content/usr_files.csv')

        install_packages()

        scan_and_write('/usr/', '/content/usr_files_new.csv')
        changed_files = compare_files('/content/usr_files.csv', '/content/usr_files_new.csv')

        with tarfile.open('/content/CachedRVC.tar.gz', 'w:gz') as new_tar:
            for file in changed_files:
                new_tar.add(file)
                print(f'Added to tar: {file}')

        os.makedirs('/content/drive/MyDrive/RVC_Cached', exist_ok=True)
        shutil.copy('/content/CachedRVC.tar.gz', '/content/drive/MyDrive/RVC_Cached/CachedRVC.tar.gz')
        print('Updated CachedRVC.tar.gz copied to Google Drive.')
        print('Dependencies fully up to date; future runs should be faster.')