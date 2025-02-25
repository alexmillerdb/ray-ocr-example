# ray-ocr-example

Run large-scale OCR pipelines using [Ray](https://www.ray.io/) on [Databricks](https://docs.databricks.com/aws/en/machine-learning/ray/).

## Prerequisites
- A Databricks workspace, configured in [`databricks.yml`](databricks.yml)  
- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) and Databricks VS Code extension (if you want to develop locally in VS Code IDE)

## Setup
1. **Clone** this repo and open it using either [Databricks Git folders](https://docs.databricks.com/aws/en/repos/) or [Databricks VS Code extension](https://docs.databricks.com/aws/en/dev-tools/vscode-ext/).  
2. **Install the Databricks extension** for VS Code if you are using VS Code.  
3. **Create Databricks cluster** add the [init.sh](scripts/init.sh) file to your cluster config (referenced in [02-ray-ocr-pipeline.py](notebooks/02-ray-ocr-pipeline.py)) to install Tesseract and other tools on cluster startup.


## Running the Demo
1. **Attach** to the configured cluster in Databricks or VS Code (ensure init.sh is added to cluster or it will not run).  
2. **Open [00-setup-env.ipynb](notebooks/00-setup-env.ipynb)** to set up the catalog, schema, and volume.  
3. **Run [01-create-dataset.py](notebooks/01-create-dataset.py)** to create a sample PDFs table (`pixparse_pdfs`).  
4. **Open [02-ray-ocr-pipeline.py](notebooks/02-ray-ocr-pipeline.py)** and run cells. This sets up a Ray cluster and processes PDFs with OCR.

## Notes
- Adjust any **Spark cluster configs** (e.g., `init_scripts`) in `02-ray-ocr-pipeline.py`.
- Use Ray Dashboard to monitor cluster performance (CPU utilization, memory, etc.), Jobs, Tasks, Actors, and more
- Check logs in the Databricks job output or Spark UI for troubleshooting.  

## Contributing
Submit pull requests or open an issue for discussions.
