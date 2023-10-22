sudo apt update -y
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa 
sudo apt install -y python3.9
sudo apt install -y python3-pip
pip install -r requirements.txt
sudo apt-get install -y libsndfile1
sudo rm -rf /usr/lib/python3/dist-packages/OpenSSL
sudo pip3 install pyopenssl
sudo pip3 install pyopenssl --upgrade

sudo apt update && sudo apt upgrade
sudo apt autoremove nvidia* --purge
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-driver-525
reboot
nvidia-smi

sudo apt update && sudo apt upgrade
sudo apt install nvidia-cuda-toolkit

sudo apt-get update -y
sudo apt-get -y install cuda

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

pip uninstall ffmpeg-python
pip uninstall ffmpeg

pip install ffmpeg-python
sudo snap install ffmpeg