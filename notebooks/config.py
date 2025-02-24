from pydantic import BaseModel, Field
from typing import Optional, Tuple

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
        description="Column name containing PDF content"
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