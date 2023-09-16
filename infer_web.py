from vc_infer_pipeline import VC
from utils import Audio
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from fairseq import checkpoint_utils
from config import Config
import logging
import torch
import numpy as np
import warnings
import traceback
import os
import shutil
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" %
              (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" %
              (now_dir), ignore_errors=True)

os.makedirs(os.path.join(now_dir, "audios"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)

config = Config()

# Determinar si existe una tarjeta N que pueda usarse para entrenar y acelerar la inferencia.
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

isinterrupted = 0

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "Lamentablemente, no tiene una tarjeta gráfica adecuada para soportar su entrenamiento"
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "weights"
index_root = "./logs/"
audio_root = "audios"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []

global indexes_list
indexes_list = []

audio_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s\\%s" % (root, name))

for root, dirs, files in os.walk(audio_root, topdown=False):
    for name in files:
        audio_paths.append("%s/%s" % (root, name))


def vc_single(
    sid,
    input_audio_path0,
    input_audio_path1,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path0 is None or input_audio_path0 is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        if input_audio_path0 == "":
            audio = Audio.load_audio(
                input_audio_path1, 16000, True, 8.0, 1.2)
        else:
            audio = Audio.load_audio(
                input_audio_path0, 16000, True, 8.0, 1.2)

        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if not hubert_model:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )

        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path1,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

# 一个选项卡全局只能有一个音色


def get_vc(model_name, to_return_protect0, to_return_protect1):
    global n_spk, tgt_sr, net_g, vc, cpt, version

    # Comprobar si se pasó uno o varios modelos
    if model_name == "" or model_name == []:
        global hubert_model
        if hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("Limpiar caché")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None

            # Si hay una GPU disponible, libera la memoria de la GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Bloque de abajo no limpia completamente
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half)
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return ({
            "success": False,
            "message": "No se proporcionó un sid"
        })

    person = "%s/%s" % (weight_root, model_name)
    print("Cargando %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(
                *cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(
                *cpt["config"], is_half=config.is_half)
    else:
        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q

    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return ({
        "protect0": 0.5 if if_f0 == 0 else to_return_protect0,
        "protect1": 0.5 if if_f0 == 0 else to_return_protect1
    })
