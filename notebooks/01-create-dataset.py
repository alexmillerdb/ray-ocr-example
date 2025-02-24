# Databricks notebook source
# MAGIC %pip install --upgrade huggingface-hub>=0.27.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00-setup-env

# COMMAND ----------

import os
# Set cache directory to UC Volume
os.environ["HF_DATASETS_CACHE"] = RAW_DATA_PATH

# COMMAND ----------

from pathlib import Path
import os
from datasets import load_dataset, Dataset

streaming_dataset = load_dataset("pixparse/pdfa-eng-wds", streaming=True, split="train")
output_dir = Path(f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/data")
os.makedirs(output_dir, exist_ok=True)

# Process in batches
batch_size = 100
for batch_idx, batch in enumerate(streaming_dataset.iter(batch_size)):
    # Save each batch as a Parquet file
    if batch_idx > 10:
        break
    batch_path = output_dir / f"batch_{batch_idx:06d}.parquet"
    Dataset.from_dict(batch).to_parquet(batch_path)