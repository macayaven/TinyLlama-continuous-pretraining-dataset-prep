import os
from torch.utils.data import DataLoader
from lightning_cloud.utils import add_s3_connection
from lightning.data import StreamingDataset, CombinedStreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from tqdm import tqdm

# Add an external S3 bucket containing some data
add_s3_connection("tinyllama-template")

# Increase by one because we need the next word as well
effective_block_size = 2048 + 1

input_dir = "/teamspace/s3_connections/tinyllama-template" # Note, you can use the data under /teamspace/datasets if you re-generated them
train_datasets = [
    StreamingDataset(
        input_dir=f"{input_dir}/slimpajama/train", # 655,350,041,334 tokens
        item_loader=TokensLoader(block_size=effective_block_size),
        shuffle=True,
        drop_last=True,
    ),
    StreamingDataset(
        input_dir=f"{input_dir}/starcoder", # 292,123,864,608 tokens
        item_loader=TokensLoader(block_size=effective_block_size),
        shuffle=True,
        drop_last=True,
    ),
]



# Mix SlimPajama data and Starcoder data with these proportions
# With those weights, we have 944,874,796,569 tokens
weights = (0.693584, 0.306416)
combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=42, weights=weights)
train_dataloader = DataLoader(combined_dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count(), prefetch_factor=10)

# Iterate over the train datasets
for batch in tqdm(train_dataloader, total=len(train_dataloader)):
    pass