# Databricks notebook source
# MAGIC %md ### Load data

# COMMAND ----------

# pdf_delta_path = 'alex_m.gen_ai.pixparse_pdfs'
# df = spark.table(pdf_delta_path)
# pdf_example = df.select("pdf").limit(1).toPandas()['pdf'][0]
# pdf_example

# COMMAND ----------

# MAGIC %md ### Process the entire PDF

# COMMAND ----------

# from pdf2image import convert_from_path, convert_from_bytes
# import pytesseract
  
# def process_pdf(pdf_binary):
#     # Convert binary PDF to images
#     pages = convert_from_bytes(pdf_binary, 300)
    
#     # Process each page sequentially
#     results = []
#     for page in pages:
#         text = pytesseract.image_to_string(page)
#         results.append(text)
#     return results

# COMMAND ----------

# # Process the binary PDF
# text_results = process_pdf(pdf_example)
# text_results

# COMMAND ----------

# len(text_results)

# COMMAND ----------

# MAGIC %md ### Split the PDF into pages and then process each page

# COMMAND ----------

# from pdf2image import convert_from_bytes
# import pytesseract
# from PIL import Image
# import io

# def split_pdf_into_pages(pdf_binary):
#     # Convert binary PDF to images - this already returns PIL Image objects
#     pages = convert_from_bytes(pdf_binary, 300)
#     return pages

# def process_single_page(page_image):
#     # Process a single PIL Image
#     text = pytesseract.image_to_string(page_image)
#     return text

# def process_pdf_by_pages(pdf_binary):
#     pages = split_pdf_into_pages(pdf_binary)
#     results = []
#     for page in pages:
#         text = process_single_page(page)
#         results.append(text)
#     return results


# COMMAND ----------

# split_test_results = process_pdf_by_pages(pdf_example)
# split_test_results

# COMMAND ----------

# MAGIC %pip install pytesseract pdf2image py-spy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a user-specific Ray cluster in a Databricks cluster: [Databricks create and connect to Ray cluster documentation](https://docs.databricks.com/en/machine-learning/ray/ray-create.html#create-a-user-specific-ray-cluster-in-a-databricks-cluster)

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
setup_ray_cluster(
  min_worker_nodes=4,       # minimum number of worker nodes to start
  max_worker_nodes=4,       # maximum number of worker nodes to start
  num_gpus_worker_node=0,   # number of GPUs to allocate per worker node
  num_gpus_head_node=0,     # number of GPUs to allocate on head node (driver)
  num_cpus_worker_node=12,  # number of CPUs to allocate on worker nodes
  num_cpus_head_node=12,    # number of CPUs to allocate on head node (driver)
  collect_log_to_path="/Volumes/alex_m/genai/pdfs/ray_collected_logs"
)


# Pass any custom configuration to ray.init
ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# COMMAND ----------

# MAGIC %md ### Create Ray Dataset using `ray.data.read_databricks_tables`

# COMMAND ----------

# MAGIC %md Create PAT then create new secret scope and add a secret (example below)
# MAGIC
# MAGIC `databricks secrets create-scope SCOPE`
# MAGIC
# MAGIC `databricks secrets put-secret SCOPE KEY`

# COMMAND ----------

import os
import ray

os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get().replace('https://','')
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("alexm", "ray-data")

pdf_delta_path = 'alex_m.gen_ai.pixparse_pdfs'

ray_dataset = ray.data.read_databricks_tables(
    warehouse_id='148ccb90800933a1',
    catalog='alex_m',
    schema='gen_ai',
    query="SELECT __key__, __url__, pdf FROM pixparse_pdfs limit 100"
)


# COMMAND ----------

ray_dataset.take(1)

# COMMAND ----------

# MAGIC %md ### Distribute OCR processor with map_batches and processing each page at a time (instead of entire PDF)

# COMMAND ----------

# MAGIC %md Feedback from Michael
# MAGIC - Ray retries for OOM during task execution
# MAGIC - Example is large data skew
# MAGIC - How to partition at the page level?
# MAGIC - Figure out DATABRICKS_HOST
# MAGIC - How to see peak memory per task or get task level information?

# COMMAND ----------

# MAGIC %md Test handling OOM Errors and move on to next PDF, implement timeout mechanism and error handling

# COMMAND ----------

import ray
from pdf2image import convert_from_bytes
import pytesseract
import time

@ray.remote
class OCRProcessor:
    """
    A Ray actor class for processing PDFs and performing OCR.
    """
    def __init__(self):
        self.tesseract = pytesseract

    def process_pdf(self, pdf_data, key, url):
        """
        Process a single PDF, converting it to images and performing OCR.

        Args:
            pdf_data (bytes or str): The PDF data to process.
            key (str): A unique identifier for the PDF.
            url (str): The URL associated with the PDF.

        Returns:
            list: A list of dictionaries containing OCR results for each page.
        """
        try:
            # Ensure pdf_data is in bytes format
            if isinstance(pdf_data, str):
                pdf_data = pdf_data.encode('utf-8')
                
            # Convert PDF to images
            pages = convert_from_bytes(pdf_data)
            
            results = []
            for page_num, page in enumerate(pages, 1):
                try:
                    # Perform OCR on the page
                    text = self.tesseract.image_to_string(page)
                    results.append({
                        "text": text if text else "", 
                        "status": "success",
                        "error": "",
                        "__key__": key if key else "",
                        "__url__": url if url else "",
                        "page_number": page_num
                    })
                except Exception as e:
                    # Handle page-level exceptions
                    results.append({
                        "text": "", 
                        "status": "error",
                        "error": str(e),
                        "__key__": key if key else "",
                        "__url__": url if url else "",
                        "page_number": page_num
                    })
            return results
        except Exception as e:
            # Handle PDF-level exceptions
            return [{
                "text": "", 
                "status": "error",
                "error": str(e),
                "__key__": key if key else "",
                "__url__": url if url else "",
                "page_number": -1
            }]

def process_batch(batch, timeout=600):
    """
    Process a batch of PDFs using the OCRProcessor.

    Args:
        batch (dict): A dictionary containing 'pdf', '__key__', and '__url__' lists.
        timeout (int): The maximum time (in seconds) allowed for processing each PDF.

    Returns:
        dict: A dictionary of lists containing the OCR results for the batch.
    """
    pdfs = batch['pdf']
    keys = batch['__key__']
    urls = batch['__url__']
    
    # Create a remote OCRProcessor instance
    processor = OCRProcessor.remote()
    
    # Submit PDF processing tasks
    futures = [processor.process_pdf.remote(pdf_data, key, url) 
               for pdf_data, key, url in zip(pdfs, keys, urls)]
    
    flattened_results = []
    for future in futures:
        try:
            # Wait for the result with a timeout
            result = ray.get(future, timeout=timeout)
            flattened_results.extend(result)
        except ray.exceptions.GetTimeoutError:
            # Handle timeout errors
            flattened_results.append({
                "text": "", 
                "status": "timeout",
                "error": "Processing timed out",
                "__key__": "",
                "__url__": "",
                "page_number": -1
            })
        except ray.exceptions.RayTaskError as e:
            # Handle Ray task errors
            flattened_results.append({
                "text": "", 
                "status": "error",
                "error": str(e),
                "__key__": "",
                "__url__": "",
                "page_number": -1
            })

    # Transform list of dictionaries into dictionary of lists
    return {
        "text": [item["text"] for item in flattened_results],
        "status": [item["status"] for item in flattened_results],
        "error": [item["error"] for item in flattened_results],
        "__key__": [item["__key__"] for item in flattened_results],
        "__url__": [item["__url__"] for item in flattened_results],
        "page_number": [item["page_number"] for item in flattened_results]
    }

# Process pages with timeout
processed_dataset = ray_dataset.map_batches(
    process_batch,
    batch_size=4,  # Process 4 PDFs at a time
    num_cpus=1,    # Allocate 1 CPU per worker
    concurrency=48 # Allow up to 60 concurrent tasks
)

# Convert the processed dataset to a Pandas DataFrame and display it
processed_dataset_pd = processed_dataset.to_pandas()
processed_dataset_pd.display()

# COMMAND ----------

ray.cluster_resources()

# COMMAND ----------

# MAGIC %md Original Code

# COMMAND ----------

import ray
from pdf2image import convert_from_bytes
import pytesseract
      
class OCRProcessor:
    def __init__(self):
        self.tesseract = pytesseract

    def __call__(self, batch):
        pdfs = batch['pdf']
        keys = batch['__key__']
        urls = batch['__url__']
        
        flattened_results = []
        for pdf_data, key, url in zip(pdfs, keys, urls):
            try:
                if isinstance(pdf_data, str):
                    pdf_data = pdf_data.encode('utf-8')
                    
                pages = convert_from_bytes(pdf_data)
                
                for page_num, page in enumerate(pages, 1):
                    try:
                        text = self.tesseract.image_to_string(page)
                        flattened_results.append({
                            "text": text if text else "", 
                            "status": "success",
                            "error": "",
                            "__key__": key if key else "",
                            "__url__": url if url else "",
                            "page_number": page_num
                        })
                    except Exception as e:
                        flattened_results.append({
                            "text": "", 
                            "status": "error",
                            "error": str(e),
                            "__key__": key if key else "",
                            "__url__": url if url else "",
                            "page_number": page_num
                        })
            except Exception as e:
                flattened_results.append({
                    "text": "", 
                    "status": "error",
                    "error": str(e),
                    "__key__": key if key else "",
                    "__url__": url if url else "",
                    "page_number": -1
                })

        # Transform list of dictionaries into dictionary of lists
        return {
            "text": [item["text"] for item in flattened_results],
            "status": [item["status"] for item in flattened_results],
            "error": [item["error"] for item in flattened_results],
            "__key__": [item["__key__"] for item in flattened_results],
            "__url__": [item["__url__"] for item in flattened_results],
            "page_number": [item["page_number"] for item in flattened_results]
        }


# Process the dataset using map_batches
processed_dataset = ray_dataset.map_batches(
    OCRProcessor,
    batch_size=4,  # Process 4 PDFs at a time
    num_cpus=1,    # Allocate 1 CPU per worker
    concurrency=48  # Number of concurrent workers
)
processed_dataset_pd = processed_dataset.to_pandas()
processed_dataset_pd.display()

# COMMAND ----------

# DBTITLE 1,Config to test processing time
import ray
import time

# Define configurations to test different CPU and concurrency combinations
configs = [
    # Test CPU scaling
    {"batch_size": 4, "num_cpus": 1, "concurrency": 16},  # Baseline
    {"batch_size": 4, "num_cpus": 2, "concurrency": 16},  # Double CPU per worker
    {"batch_size": 4, "num_cpus": 3, "concurrency": 16},  # Quad CPU per worker
    
    # Test concurrency scaling
    # {"batch_size": 4, "num_cpus": 1, "concurrency": 8},   # Half concurrency
    {"batch_size": 4, "num_cpus": 1, "concurrency": 32},  # Double concurrency
    {"batch_size": 4, "num_cpus": 1, "concurrency": 48},  # Max concurrency
    
    # Test batch size impact
    {"batch_size": 8, "num_cpus": 1, "concurrency": 16},  # Larger batch
    {"batch_size": 2, "num_cpus": 1, "concurrency": 16},   # Smaller batch
    {"batch_size": 8, "num_cpus": 2, "concurrency": 16}
]

results = []
for config in configs:
    print(config)
    start_time = time.time()
    processed_dataset = ray_dataset.map_batches(
        OCRProcessor,
        batch_size=config["batch_size"],
        num_cpus=config["num_cpus"],
        concurrency=config["concurrency"]
    )
    processed_dataset.materialize()  # Force execution
    end_time = time.time()
    
    results.append({
        **config,
        "time_taken": end_time - start_time
    })

# Sort and print results by execution time
results = sorted(results, key=lambda x: x["time_taken"])
results

# COMMAND ----------

import pandas as pd

results_df = pd.DataFrame(results)
results_df['experiment'] = [i for i in range(len(results))]
display(results_df)

# COMMAND ----------

ray_dataset = ray.data.read_databricks_tables(
    warehouse_id='d1184b8c2a8a87eb',
    catalog='alex_m',
    schema='gen_ai',
    query="SELECT __key__, __url__, pdf FROM pixparse_pdfs limit 1000",
    # override_num_blocks=16  # Increase number of blocks for better parallelism
)

best_config = results_df.sort_values('time_taken').iloc[0]

processed_dataset = ray_dataset.map_batches(
        OCRProcessor,
        batch_size=int(best_config.batch_size),
        num_cpus=int(best_config.num_cpus),
        concurrency=int(best_config.concurrency)
    )

processed_data = processed_dataset.to_pandas()
display(processed_data)

# COMMAND ----------

ray.cluster_resources()

# COMMAND ----------

# MAGIC %md ### Time to run
# MAGIC - 4 minutes for 100 rows

# COMMAND ----------

processed_data = processed_dataset.to_pandas()
display(processed_data)
