import os
from time import time
from lightning_sdk import Studio, Machine

HF_TOKEN = "PUT_YOUR_OWN_TOKEN"
assert HF_TOKEN != "PUT_YOUR_OWN_TOKEN", "Provide your own token"

studio = Studio(name="StarCoder_Dataset")
studio.start(machine=Machine.DATA_PREP)

t0 = time()
studio.run("sudo apt-get install git-lfs")
studio.run("pip install -U 'huggingface_hub[cli]'")
studio.run("git config --global credential.helper store")
studio.run(f"huggingface-cli login --token {HF_TOKEN} --add-to-git-credential")
studio.run("git clone https://huggingface.co/datasets/bigcode/starcoderdata data")
print(time() - t0)

studio.stop()