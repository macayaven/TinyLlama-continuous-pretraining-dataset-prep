import json
from pathlib import Path
import zstandard as zstd
from litdata import optimize
from tokenizer import Tokenizer
from functools import partial
from lightning_sdk import Machine

# 1. Function to tokenize the text contained within the Slimpajama files
def tokenize_fn(filepath, tokenizer=None):
    with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            text = json.loads(row)["text"]
            if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                continue  # exclude the GitHub data since it overlaps with starcoder
            text_ids = tokenizer.encode(text, bos=False, eos=True)
            yield text_ids

# 2. Generate the inputs (we are going to optimize all the compressed json files from SlimPajama dataset)
input_dir = "/teamspace/s3_connections/tinyllama-template/SlimPajama-627B/train"# "/teamspace/studios/SlimPajama_Dataset/data/train"
inputs = [str(file) for file in Path(input_dir).rglob("*.jsonl.zst")]

# 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
outputs = optimize(
    fn=partial(tokenize_fn, tokenizer=Tokenizer("./checkpoints/Llama-2-7b-hf")), # Note: You can use HF tokenizer or any others
    inputs=inputs,
    output_dir="/teamspace/datasets/slimpajama/train",
    chunk_size=(2049 * 8012),
    num_nodes=16,
    machine=Machine.DATA_PREP, # use 32 CPU machine
    reorder_files=False,
)