# Databricks notebook source
# MAGIC %md
# MAGIC # Extract Text from PDFs using Tesseract OCR and Ray
# MAGIC
# MAGIC This notebook extracts texts from a UC Volume full of PDFs, leveraging Ray to scale Tesseract for optical character recognition (OCR). It demonstrates how to properly configure a "Ray on Databricks" cluster, and use Ray alongside Spark (on the same cluster) for more intensive data processing. 
# MAGIC
# MAGIC ## Requirements: 
# MAGIC
# MAGIC Before proceeding, you need:
# MAGIC * A classic compute cluster with the following configurations (see bottom of notebook for example):
# MAGIC   * Databricks Runtime: 15.4 LTS ML
# MAGIC   * Node Types: 16 core workers (instance type will vary by cloud)
# MAGIC   * Number of Workers: 6 (not autoscaling)
# MAGIC   * Access Mode: Single-User Dedicated
# MAGIC   * Add Spark conf for Spark to Ray conversion ([AWS](https://docs.databricks.com/aws/en/spark/conf)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/spark/conf)|[GCP](https://docs.databricks.com/gcp/en/spark/conf)): `"spark.databricks.pyspark.dataFrameChunk.enabled": "true"` 
# MAGIC
# MAGIC * Note: As of March 2025, Ray on Databricks does not run on Serverless compute (either notebook or jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC Before running the main script, we must first install our OCR library (Tesseract) on all nodes of our cluster. 
# MAGIC
# MAGIC To do this, we will use the Python `subprocess` command. The script below can be adapted to any other dependency; alternatively, you can install cluster-wide non-Python dependencies via [init scripts](https://docs.databricks.com/aws/en/init-scripts/):

# COMMAND ----------

from typing import List
def install_apt_get_packages(package_list: List[str]):
    """
    Installs apt-get packages required by the parser. 
    Source: https://github.com/databricks/genai-cookbook/

    Parameters:
        package_list (str): A space-separated list of apt-get packages.
    """
    import subprocess

    num_workers = max(1, int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")))

    packages_str = " ".join(package_list)
    command = f"sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install {packages_str} -y"
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    data = spark.sparkContext.parallelize(range(num_workers), num_workers)
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect()
        print(f"{package_list} libraries installed")
    except Exception as e:
        print(f"Couldn't install {package_list} on all nodes: {e}")
        raise e

install_apt_get_packages(["poppler-utils", "tesseract-ocr", "libmagic-dev"])

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we can `pip install` Python dependencies, then restart Python.

# COMMAND ----------

# MAGIC %pip install pytesseract pdf2image
# MAGIC %pip install --upgrade huggingface-hub>=0.27.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. Environment setup
# MAGIC
# MAGIC **Update the following variables** with [Unity Catalog]() details for your particular environment. The code below will create the locations if they do not exist.
# MAGIC
# MAGIC - `UC_CATALOG`: The catalog name in Unity Catalog.
# MAGIC - `UC_SCHEMA`: The schema name within the catalog.
# MAGIC - `UC_VOLUME`: The volume name within the schema.
# MAGIC
# MAGIC You must have `USE_CATALOG` privilege on the catalog and `USE_SCHEMA` and `CREATE_TABLE` privileges on the schema.

# COMMAND ----------

UC_CATALOG = "shared"
UC_SCHEMA = "tjc_ray"
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

import os
os.environ["RAY_UC_VOLUMES_FUSE_TEMP_DIR"] = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/temp_data"

# COMMAND ----------

# MAGIC %md
# MAGIC Run environment checks before proceeding.

# COMMAND ----------

assert os.environ.get("RAY_UC_VOLUMES_FUSE_TEMP_DIR") is not None, "Environment variable RAY_UC_VOLUMES_FUSE_TEMP_DIR is not set"
assert spark.conf.get("spark.databricks.pyspark.dataFrameChunk.enabled") == "true", "Spark conf 'spark.databricks.pyspark.dataFrameChunk.enabled' is not set to 'true'. Set this in the cluster config and restart the cluster before proceeding."

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2. Create Dataset
# MAGIC Download sample PDF documents from Hugging Face. 
# MAGIC
# MAGIC - Sets up HuggingFace dataset cache location
# MAGIC - Downloads sample PDFs from the pixparse/pdfa-eng-wds dataset
# MAGIC - Converts the downloaded batches to parquet format
# MAGIC - Creates a Delta table from the parquet files
# MAGIC
# MAGIC This cell will take a few minutes to run.

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
        break # For demonstration purposes, only download a few batches
    batch_path = output_dir / Path(f"batch_{batch_idx:06d}.parquet")
    Dataset.from_dict(batch).to_parquet(batch_path)

delta_table_name = f"{UC_CATALOG}.{UC_SCHEMA}.pixparse_pdfs"
df = spark.read.parquet(f"{output_dir}/*.parquet")
df.write.mode("overwrite").saveAsTable(delta_table_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Ray OCR Pipeline
# MAGIC
# MAGIC Process PDFs with Distributed OCR using Ray Data `map_batches`
# MAGIC
# MAGIC - Implements a distributed OCR pipeline using Ray
# MAGIC - PDFProcessor: Converts PDF documents to images using pdf2image
# MAGIC - OCRProcessor: Performs OCR on images using pytesseract
# MAGIC - Configurable batch processing with concurrent execution
# MAGIC - Stores results in a UC Delta table with extracted text and metadata
# MAGIC
# MAGIC This example also follows best practices for configuration values for both data and Ray setup, leveraging `pydantic` [models](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel). 

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
    

# COMMAND ----------

def main(config: ProcessingConfig) -> None:
    # Read in binary PDF data from UC table
    sdf = (
        spark.read.table(config.data.input_table)
        .select(
            F.col(config.data.pdf_column).alias("content"), 
            F.col(config.data.path_column).alias("path")
        )
    )
    if config.data.limit_rows:
        sdf = sdf.limit(config.data.limit_rows)

    # Convert data from Spark to Ray. 
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

    # Set Ray cluster configuration
    ray_context = setup_ray_cluster(
        min_worker_nodes=4,
        max_worker_nodes=4,
        num_gpus_worker_node=0,
        num_gpus_head_node=0,
        num_cpus_worker_node=12, #Since we are using Spark+Ray together, make sure this number is smaller than total cores per worker.
        num_cpus_head_node=8,
    )
    # Call Ray init to start cluster
    ray.init(ignore_reinit_error=True)

    # First, convert PDFs to images
    pages_dataset = ray_dataset.map_batches(
        PDFProcessor,
        batch_size=config.ray.pdf_batch_size,
        num_cpus=config.ray.pdf_num_cpus,
        concurrency=config.ray.pdf_concurrency,
    )

    # Second, perform OCR on images
    ocr_dataset = pages_dataset.map_batches(
        OCRProcessor,
        batch_size=config.ray.ocr_batch_size,
        num_cpus=config.ray.ocr_num_cpus,
        concurrency=config.ray.ocr_concurrency,
    )

    # Write final dataset to UC
    # Run help(ray.data.Dataset.write_databricks_table) for optional params
    _ = ray.data.Dataset.write_databricks_table(
      ray_dataset = ocr_dataset,
      name = config.data.output_table,
      mode = "overwrite"
    )


# COMMAND ----------

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

# Start OCR pipeline with above config
main(config)

# After running this cell, you should see a hyperlink below: `Open Ray Cluster Dashboard in a new tab`
# Read more about the Ray Dashboard here: https://docs.ray.io/en/latest/ray-observability/getting-started.html

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Review Results
# MAGIC
# MAGIC Let's use Spark to read in a few of the resulting rows. As you can see, we've successfully extracted text from the original PDFs, and done it at massive scale!

# COMMAND ----------


print(f"The final table contains {spark.read.table(output_table).count()} rows")

display(spark.read.table(output_table).limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Cleanup
# MAGIC
# MAGIC Be sure to terminate your Ray cluster after the job completes - otherwise, the cluster will keep running. 
# MAGIC
# MAGIC Uncomment and run the cell below.

# COMMAND ----------

# shutdown_ray_cluster()
# ray.shutdown()

# COMMAND ----------

# MAGIC %md
# MAGIC This code has been tested with the following cluster configuration. 
# MAGIC * Databricks Runtime: 15.4 LTS ML
# MAGIC * Node Types: 16 core workers (instance type will vary by cloud)
# MAGIC * Number of Workers: 6 (not autoscaling)
# MAGIC * Access Mode: Single-User Dedicated
# MAGIC
# MAGIC Here is an example Cluster Spec (JSON) - this can be used to create a new cluster via [API](https://docs.databricks.com/api/workspace/clusters/create), or to match settings in the UI.
# MAGIC ```
# MAGIC "spec": {
# MAGIC         "cluster_name": "ray_on_databricks_cluster",
# MAGIC         "spark_version": "15.4.x-scala2.12",
# MAGIC         "spark_conf": {
# MAGIC             "spark.databricks.pyspark.dataFrameChunk.enabled": "true"
# MAGIC         },
# MAGIC         "node_type_id": "r5d.4xlarge",
# MAGIC         "autotermination_minutes": 120,
# MAGIC         "single_user_name": "...",
# MAGIC         "data_security_mode": "DATA_SECURITY_MODE_AUTO",
# MAGIC         "runtime_engine": "STANDARD",
# MAGIC         "kind": "CLASSIC_PREVIEW",
# MAGIC         "use_ml_runtime": true,
# MAGIC         "is_single_node": false,
# MAGIC         "num_workers": 6
# MAGIC     }
# MAGIC ```
# MAGIC
# MAGIC Please note that further cluster tuning may be required to apply the concepts here to your own data. For help, see: [Scale Ray clusters on Databricks
# MAGIC ](https://docs.databricks.com/aws/en/machine-learning/ray/scale-ray)
