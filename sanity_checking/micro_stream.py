import os
from torch.utils.data import DataLoader
from lightning_cloud.utils import add_s3_connection
from lightning.data import StreamingDataset, CombinedStreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from tqdm import tqdm

# Increase by one because we need the next word as well
effective_block_size = 2048 + 1

dataset = StreamingDataset(
    input_dir="output_dir", # micro example
    item_loader=TokensLoader(block_size=effective_block_size),
    shuffle=True,
    drop_last=True,
)
train_dataloader = DataLoader(dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count(), prefetch_factor=10)

# Iterate over the train datasets
for batch in tqdm(train_dataloader, total=len(train_dataloader)):
    pass