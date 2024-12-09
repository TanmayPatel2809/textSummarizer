from src import logger
from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.pipeline.data_validation_pipeline import DataValidationTrainingPipeline



STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.initiate_data_ingestion()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Validation Stage"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} Started <<<<<")
    obj = DataValidationTrainingPipeline()
    obj.initiate_data_validation()
    logger.info(f">>>>> Stage {STAGE_NAME} Completed <<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e
