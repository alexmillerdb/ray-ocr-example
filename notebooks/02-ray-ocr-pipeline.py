# Databricks notebook source
# MAGIC %pip install pytesseract pdf2image
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./00-setup-env

# COMMAND ----------
# MAGIC %md
# MAGIC ### Cluster Config to use for reproducibility:

# MAGIC ```json
# MAGIC {
# MAGIC     "cluster_name": "alex miller OCR cluster udf",
# MAGIC     "spark_version": "15.4.x-scala2.12",
# MAGIC     "spark_conf": {
# MAGIC         "spark.databricks.pyspark.dataFrameChunk.enabled": "true"
# MAGIC     },
# MAGIC     "azure_attributes": {
# MAGIC         "availability": "ON_DEMAND_AZURE"
# MAGIC     },
# MAGIC     "node_type_id": "Standard_D16ads_v5",
# MAGIC     "autotermination_minutes": 120,
# MAGIC     "init_scripts": [
# MAGIC         {
# MAGIC             "workspace": {
# MAGIC                 "destination": "/Users/alex.miller@databricks.com/ray-ocr/init.sh"
# MAGIC             }
# MAGIC         }
# MAGIC     ],
# MAGIC     "single_user_name": "alex.miller@databricks.com",
# MAGIC     "data_security_mode": "DATA_SECURITY_MODE_AUTO",
# MAGIC     "runtime_engine": "PHOTON",
# MAGIC     "kind": "CLASSIC_PREVIEW",
# MAGIC     "use_ml_runtime": true,
# MAGIC     "is_single_node": false,
# MAGIC     "num_workers": 6,
# MAGIC     "apply_policy_default_values": false
# MAGIC }
# MAGIC ```

# COMMAND ----------
# MAGIC %md ### Setup Ray cluster:
# MAGIC - init.sh is included in Databricks cluster to download tesseract-ocr package to all nodes
# MAGIC - Spark Cluster has 6 `num_workers` but will pass 4 to Ray and leave 2 for Spark (let Spark handle to read and write process)
# MAGIC - Supplying Ray with 4 `min_worker_nodes` and `max_worker_nodes` (leaving 2 for Spark)

# COMMAND ----------
from ray.util.spark import setup_ray_cluster, MAX_NUM_WORKER_NODES, shutdown_ray_cluster
import ray

restart = True
if restart is True:
  try:
    shutdown_ray_cluster()
  except:
    pass
  try:
    ray.shutdown()
  except:
    pass

# Ray allows you to define custom cluster configurations using setup_ray_cluster function
# This allows you to allocate CPUs and GPUs on Ray cluster
ray_context = setup_ray_cluster(
  min_worker_nodes=4,       # minimum number of worker nodes to start
  max_worker_nodes=4,       # maximum number of worker nodes to start (autoscaling)
  num_gpus_worker_node=0,   # number of GPUs to allocate per worker node
  num_gpus_head_node=0,     # number of GPUs to allocate on head node (driver)
  num_cpus_worker_node=12,   # number of CPUs to allocate on worker nodes, only giving Ray 1 and Spark the rest
  num_cpus_head_node=8,    # number of CPUs to allocate on head node (driver)
#   collect_log_tp_path="/Volumes/alex_m/gen_ai/pdfs/ray_collected_logs"
)

# Pass any custom configuration to ray.init
ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# COMMAND ----------
# MAGIC %md ### Description of the Code
# MAGIC The code below is designed to process PDF documents using a combination of Spark and Ray for distributed computing. The workflow involves the following steps:
# MAGIC
# MAGIC 1. **Cluster Setup**: The Ray cluster is set up with specific configurations for the number of worker nodes, CPUs, and GPUs allocated to both the head node and worker nodes. This setup ensures efficient resource utilization between Spark and Ray.
# MAGIC 
# MAGIC 2. **PDF Processing**: A `PDFProcessor` class is defined to convert PDF documents into images. The class includes methods to convert PDF data to image bytes and handle batches of PDF documents.
# MAGIC 
# MAGIC 3. **OCR Processing**: An `OCRProcessor` class is defined to perform Optical Character Recognition (OCR) on the images generated from the PDF documents. The class includes methods to convert image bytes to PIL images and handle batches of images for OCR processing.

# MAGIC 4. **Main Function**: The `main` function orchestrates the entire workflow:
# MAGIC     - Reads PDF documents from a Spark table.
# MAGIC     - Converts the Spark DataFrame to a Ray Dataset.
# MAGIC     - Processes the PDFs to convert them into images using the `PDFProcessor`.
# MAGIC     - Performs OCR on the images using the `OCRProcessor`.
# MAGIC     - Converts the OCR results to a Pandas DataFrame and displays the results.
# MAGIC     - Saves the processed data back to a Spark table.
# MAGIC
# MAGIC 5. **Shutdown**: Finally, the Ray cluster is shut down to release the resources.
# MAGIC 
# MAGIC This approach leverages the parallel processing capabilities of Ray and the distributed data handling capabilities of Spark to efficiently process large volumes of PDF documents and extract text using OCR.

# COMMAND ----------
import io
from typing import List, Dict, Any
import ray
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import time
from pyspark.sql import functions as F
from config import ProcessingConfig, RayConfig, DataConfig


class PDFProcessor:
    """
    A class for processing PDF documents and converting them to images.
    """
    def __init__(self):
        pass

    @staticmethod
    def pdf_to_image_bytes(pdf_data: bytes) -> List[bytes]:
        """
        Convert PDF data to a list of image byte strings.

        Args:
            pdf_data (bytes): Raw PDF data.

        Returns:
            List[bytes]: List of image byte strings.
        """
        pages = convert_from_bytes(pdf_data)
        return [PDFProcessor._image_to_bytes(page) for page in pages]

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        """
        Convert a PIL Image to bytes.

        Args:
            image (Image.Image): PIL Image object.

        Returns:
            bytes: Byte string representation of the image.
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Process a batch of PDF documents.

        Args:
            batch (Dict[str, Any]): Batch of PDF documents.

        Returns:
            Dict[str, List[Any]]: Processed batch with pages, paths, and page numbers.
        """
        results = []
        for content, path in zip(batch["content"], batch["path"]):
            try:
                pages = PDFProcessor.pdf_to_image_bytes(content)
                results.extend([
                    {"page": page, "path": path, "page_number": i + 1}
                    for i, page in enumerate(pages)
                ])
            except Exception as e:
                results.append({
                    "page": b"",
                    "path": path,
                    "page_number": -1,
                    "error": str(e)
                })

        return {
            "page": [item["page"] for item in results],
            "path": [item["path"] for item in results],
            "page_number": [item["page_number"] for item in results],
            "error": [item.get("error", "") for item in results]
        }

class OCRProcessor:
    """
    A class for performing OCR on images.
    """

    def __init__(self):
        self.tesseract = pytesseract

    @staticmethod
    def bytes_to_pil(image_bytes: bytes) -> Image.Image:
        """
        Convert image bytes to PIL Image.

        Args:
            image_bytes (bytes): Byte string representation of an image.

        Returns:
            Image.Image: PIL Image object.
        """
        return Image.open(io.BytesIO(image_bytes))

    # def process_batch(self, batch: Dict[str, Any]) -> Dict[str, List[Any]]:
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Process a batch of images with OCR.

        Args:
            batch (Dict[str, Any]): Batch of images.

        Returns:
            Dict[str, List[Any]]: OCR results for the batch.
        """
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

        return {key: [item[key] for item in results] for key in results[0]}

def main(config: ProcessingConfig = ProcessingConfig()) -> None:
    """
    Main processing pipeline for PDF OCR.
    
    Args:
        config: ProcessingConfig object containing all processing parameters
    """
    # Read PDFs from Spark table
    sdf = (
        spark.read.table(config.data.input_table)
        .select(
            F.col(config.data.pdf_column).alias("content"), 
            F.col(config.data.path_column).alias("path")
        )
    )
    
    if config.data.limit_rows:
        sdf = sdf.limit(config.data.limit_rows)

    # Create Ray Dataset
    ray_dataset = ray.data.from_spark(sdf)

    # Set concurrency based on available CPUs
    cpu_count = ray.cluster_resources().get("CPU", 1)
    min_concurrency = int(cpu_count * config.ray.min_concurrency_factor)
    max_concurrency = int(cpu_count * config.ray.max_concurrency_factor)

    # Process PDFs
    pages_dataset = ray_dataset.map_batches(
        PDFProcessor,
        batch_size=config.ray.pdf_batch_size,
        num_cpus=config.ray.pdf_num_cpus,
        concurrency=config.ray.pdf_concurrency,
    )

    # Perform OCR
    ocr_dataset = pages_dataset.map_batches(
        OCRProcessor,
        batch_size=config.ray.ocr_batch_size,
        num_cpus=config.ray.ocr_num_cpus,
        concurrency=config.ray.ocr_concurrency,
    )

    # Convert to pandas and display results
    ocr_dataset_pd = ocr_dataset.to_pandas()
    ocr_dataset_pd.display()

    # Save results
    processed_spark_df = spark.createDataFrame(ocr_dataset_pd)
    processed_spark_df.write.mode("overwrite").saveAsTable(config.data.output_table)


input_table = f"{UC_CATALOG}.{UC_SCHEMA}.pixparse_pdfs"
output_table = f"{UC_CATALOG}.{UC_SCHEMA}.ray_ocr"

# Example usage with custom configuration
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