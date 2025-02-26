# Databricks notebook source
# MAGIC %pip install pytesseract pdf2image
# MAGIC %pip install --upgrade huggingface-hub>=0.27.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ###
# MAGIC 1. Environment setup: update the following variables

# MAGIC - `UC_CATALOG`: The catalog name in Unity Catalog.
# MAGIC - `UC_SCHEMA`: The schema name within the catalog.
# MAGIC - `UC_VOLUME`: The volume name within the schema.
# MAGIC - `RAW_DATA_PATH`: The path where raw data will be stored.
# COMMAND ----------
UC_CATALOG = "alex_m"
UC_SCHEMA = "ocr"
UC_VOLUME = "ocr_data"
RAW_DATA_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/raw_data"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {UC_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}")
dbutils.fs.mkdirs(RAW_DATA_PATH)

print(f"UC catalog: {UC_CATALOG}")
print(f"UC schema: {UC_SCHEMA}")
print(f"UC volume: {UC_VOLUME}")
print(f"Raw data path: {RAW_DATA_PATH}")

# COMMAND ----------
# MAGIC %md ###
# MAGIC 2. Create Dataset: Download sample PDF documents from Hugging Face

# MAGIC - Sets up HuggingFace dataset cache location
# MAGIC - Downloads sample PDFs from the pixparse/pdfa-eng-wds dataset
# MAGIC - Converts the downloaded batches to parquet format
# MAGIC - Creates a Delta table from the parquet files
# COMMAND ----------
import os
os.environ["HF_DATASETS_CACHE"] = RAW_DATA_PATH

from pathlib import Path
from datasets import load_dataset, Dataset

output_dir = Path(f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/data")
os.makedirs(output_dir, exist_ok=True)

streaming_dataset = load_dataset("pixparse/pdfa-eng-wds", streaming=True, split="train")
batch_size = 100

for batch_idx, batch in enumerate(streaming_dataset.iter(batch_size)):
    if batch_idx > 10:
        break
    batch_path = output_dir / Path(f"batch_{batch_idx:06d}.parquet")
    Dataset.from_dict(batch).to_parquet(batch_path)

delta_table_name = f"{UC_CATALOG}.{UC_SCHEMA}.pixparse_pdfs"
df = spark.read.parquet(f"{output_dir}/*.parquet")
df.write.mode("overwrite").saveAsTable(delta_table_name)

# COMMAND ----------
# MAGIC %md ###
# MAGIC 3. Ray OCR Pipeline: Process PDFs with Distributed OCR using Ray Data `map_batches`

# MAGIC - Implements a distributed OCR pipeline using Ray
# MAGIC - PDFProcessor: Converts PDF documents to images using pdf2image
# MAGIC - OCRProcessor: Performs OCR on images using pytesseract
# MAGIC - Configurable batch processing with concurrent execution
# MAGIC - Stores results in a Delta table with extracted text and metadata

# COMMAND ----------
import io
import time
from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Dict, Any

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

from pyspark.sql import functions as F
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster


class RayConfig(BaseModel):
    min_concurrency_factor: float = Field(
        default=0.4,
        description="Factor to multiply CPU count for min concurrency"
    )
    max_concurrency_factor: float = Field(
        default=0.9,
        description="Factor to multiply CPU count for max concurrency"
    )
    pdf_batch_size: int = Field(
        default=100,
        description="Batch size for PDF processing"
    )
    pdf_num_cpus: int = Field(
        default=1,
        description="Number of CPUs per PDF processing task"
    )
    pdf_concurrency: Tuple[int, int] = Field(
        default=(15, 28),
        description="Min and max concurrency for PDF processing"
    )
    ocr_batch_size: int = Field(
        default=8,
        description="Batch size for OCR processing"
    )
    ocr_num_cpus: int = Field(
        default=1,
        description="Number of CPUs per OCR task"
    )
    ocr_concurrency: Tuple[int, int] = Field(
        default=(20, 28),
        description="Min and max concurrency for OCR processing"
    )

class DataConfig(BaseModel):
    input_table: str = Field(
        default="alex_m.gen_ai.pixparse_pdfs",
        description="Input table containing PDFs"
    )
    output_table: str = Field(
        default="alex_m.gen_ai.ray_ocr",
        description="Output table for OCR results"
    )
    pdf_column: str = Field(
        default="pdf",
        description="Column name containing PDF content in binary format"
    )
    path_column: str = Field(
        default="__url__",
        description="Column name containing PDF paths"
    )
    limit_rows: Optional[int] = Field(
        default=1000,
        description="Limit number of rows to process (None for all)"
    )

class ProcessingConfig(BaseModel):
    ray: RayConfig = Field(default_factory=RayConfig)
    data: DataConfig = Field(default_factory=DataConfig)

class PDFProcessor:
    def __init__(self):
        pass

    @staticmethod
    def pdf_to_image_bytes(pdf_data: bytes) -> List[bytes]:
        pages = convert_from_bytes(pdf_data)
        return [PDFProcessor._image_to_bytes(page) for page in pages]

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, List[Any]]:
        results = []
        for content, path in zip(batch["content"], batch["path"]):
            try:
                pages = self.pdf_to_image_bytes(content)
                for idx, page in enumerate(pages):
                    results.append({
                        "page": page,
                        "path": path,
                        "page_number": idx
                    })
            except Exception as e:
                results.append({
                    "page": b"",
                    "path": path,
                    "page_number": 0,
                    "error": str(e)
                })

        return {
            "page": [item["page"] for item in results],
            "path": [item["path"] for item in results],
            "page_number": [item["page_number"] for item in results],
            "error": [item.get("error", "") for item in results]
        }

class OCRProcessor:
    def __init__(self):
        self.tesseract = pytesseract

    @staticmethod
    def bytes_to_pil(image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes))

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, List[Any]]:
        results = []
        for page, path, page_number in zip(batch["page"], batch["path"], batch["page_number"]):
            start_time = time.time()
            try:
                image = self.bytes_to_pil(page)
                text = self.tesseract.image_to_string(image)
                results.append({
                    "text": text or "",
                    "status": "success",
                    "error": "",
                    "path": path,
                    "page_number": page_number,
                    "duration": time.time() - start_time
                })
                del image
            except Exception as e:
                results.append({
                    "text": "",
                    "status": "error",
                    "error": str(e),
                    "path": path,
                    "page_number": page_number,
                    "duration": time.time() - start_time
                })

        return {key: [item[key] for item in results] for key in results[0].keys()}
    

def main(config: ProcessingConfig) -> None:
    sdf = (
        spark.read.table(config.data.input_table)
        .select(
            F.col(config.data.pdf_column).alias("content"), 
            F.col(config.data.path_column).alias("path")
        )
    )
    if config.data.limit_rows:
        sdf = sdf.limit(config.data.limit_rows)

    ray_dataset = ray.data.from_spark(sdf)

    # Initialize or restart Ray cluster
    try:
        shutdown_ray_cluster()
    except:
        pass
    try:
        ray.shutdown()
    except:
        pass

    ray_context = setup_ray_cluster(
        min_worker_nodes=4,
        max_worker_nodes=4,
        num_gpus_worker_node=0,
        num_gpus_head_node=0,
        num_cpus_worker_node=12,
        num_cpus_head_node=8,
    )
    ray.init(ignore_reinit_error=True)

    pages_dataset = ray_dataset.map_batches(
        PDFProcessor,
        batch_size=config.ray.pdf_batch_size,
        num_cpus=config.ray.pdf_num_cpus,
        concurrency=config.ray.pdf_concurrency,
    )

    ocr_dataset = pages_dataset.map_batches(
        OCRProcessor,
        batch_size=config.ray.ocr_batch_size,
        num_cpus=config.ray.ocr_num_cpus,
        concurrency=config.ray.ocr_concurrency,
    )

    ocr_dataset_pd = ocr_dataset.to_pandas()
    display(ocr_dataset_pd)

    # write processed dataset to Delta table
    processed_spark_df = spark.createDataFrame(ocr_dataset_pd)
    processed_spark_df.write.mode("overwrite").saveAsTable(config.data.output_table)


input_table = f"{UC_CATALOG}.{UC_SCHEMA}.pixparse_pdfs"
output_table = f"{UC_CATALOG}.{UC_SCHEMA}.ray_ocr"

config = ProcessingConfig(
    ray=RayConfig(
        pdf_batch_size=100,
        pdf_num_cpus=1,
        pdf_concurrency=(20, 28),
        ocr_batch_size=8,
        ocr_num_cpus=1,
        ocr_concurrency=(20, 28)
    ),
    data=DataConfig(
        limit_rows=1000,
        input_table=input_table,
        pdf_column="pdf",
        path_column="__url__",
        output_table=output_table
    )
)
main(config)

shutdown_ray_cluster()
ray.shutdown()