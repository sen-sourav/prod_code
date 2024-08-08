wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

conda create --name accmusic python=3.8
conda activate accmusic

# Create repos directory
mkdir -p repos

# Clone repositories and install dependencies
git clone https://github.com/sen-sourav/Ultimate-Accompaniment-Transformer.git repos/Ultimate-Accompaniment-Transformer
python3 -m pip install ./repos/Ultimate-Accompaniment-Transformer
git clone https://github.com/sen-sourav/basic-pitch-torch.git repos/basic-pitch-torch
python3 -m pip install ./repos/basic-pitch-torch
python3 -m pip install huggingface_hub
python3 -m pip install einops
python3 -m pip install torch
python3 -m pip install torch-summary
python3 -m pip install torchaudio
python3 -m pip install tqdm

# Install fluidsynth via apt
sudo apt update
sudo apt install -y fluidsynth


# Install for discord
python3 -m pip install discord
python3 -m pip install responses
