import requests
import os
import ffmpeg
import numpy as np
import random
import shutil
import torchaudio
from pydub import AudioSegment
import tempfile

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
    def load_audio(cls, file, sr):
        try:
            file = file.strip(' "\n')  # Eliminar espacios y comillas del nombre del archivo
            # Convertir a formato WAV si no lo está
            if not file.endswith(".wav"):
                file_formanted = f"{file}.wav"
                if not os.path.isfile(file_formanted):
                    (
                        ffmpeg.input(file)
                        .output(file_formanted, format="wav")
                        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
                    )
            else:
                file_formanted = file

            # Cargar el archivo formateado y devolverlo como NumPy array
            out, _ = (
                ffmpeg.input(file_formanted)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )

            # Eliminar el archivo formateado
            os.remove(file_formanted)    
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

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


def delete_files(paths):
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            if os.path.isfile(path):
                os.remove(path)