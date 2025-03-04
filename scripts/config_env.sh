
git submodule update --init --recursive

python3 -m pip install -r requirements.txt
python3 -m pip install pynini

yes | sudo apt-get install sox libsox-dev

mkdir -p pretrained_models

python3 -m pip install hf_transfer

huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir pretrained_models/CosyVoice2-0.5B --repo-type model
