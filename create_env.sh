wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda create --name accmusic python=3.8
conda activate accmusic

# Create repos directory
mkdir -p repos

# Clone repositories and install dependencies
git clone https://github.com/sen-sourav/Ultimate-Accompaniment-Transformer.git repos/Ultimate-Accompaniment-Transformer
pip install ./repos/Ultimate-Accompaniment-Transformer
git clone https://github.com/sen-sourav/basic-pitch-torch.git repos/basic-pitch-torch
pip install ./repos/basic-pitch-torch
pip install huggingface_hub
pip install einops
pip install torch-summary

# Install fluidsynth via apt
sudo apt update
sudo apt install -y fluidsynth
