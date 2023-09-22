import requests
import os
import ffmpeg
import numpy as np
import random

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
    
