# Databricks notebook source
# MAGIC %pip install --upgrade huggingface-hub>=0.27.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
# Set cache directory to UC Volume
os.environ["HF_DATASETS_CACHE"] = "/Volumes/alex_m/gen_ai/pdfs/"

# COMMAND ----------

from pathlib import Path
import os
from datasets import load_dataset, Dataset

streaming_dataset = load_dataset("pixparse/pdfa-eng-wds", streaming=True, split="train")
output_dir = Path("/Volumes/alex_m/gen_ai/pdfs/data")
os.makedirs(output_dir, exist_ok=True)

# Process in batches
batch_size = 100
for batch_idx, batch in enumerate(streaming_dataset.iter(batch_size)):
    # Save each batch as a Parquet file
    # if batch_idx > 2:
    #     break
    batch_path = output_dir / f"batch_{batch_idx:06d}.parquet"
    Dataset.from_dict(batch).to_parquet(batch_path)


# COMMAND ----------

# Dataset.from_dict(batch).to_pandas()
