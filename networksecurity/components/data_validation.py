from networksecurity.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifact
)
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import (
    read_yaml_file,
    write_yaml_file
)

from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os
import sys


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifacts,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            dataframe = pd.read_csv(file_path)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_schema_columns(self) -> list:
        """
        Extract expected column names from schema.yaml.
        """
        try:
            schema_columns = self._schema_config.get("columns", [])

            expected_columns = [
                column_name
                for column in schema_columns
                for column_name in column.keys()
            ]

            return expected_columns

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(
        self,
        dataframe: pd.DataFrame
    ) -> bool:
        """
        Validate column count and column names.
        """
        try:
            expected_columns = self.get_schema_columns()
            actual_columns = dataframe.columns.tolist()

            logging.info(
                f"Expected number of columns: {len(expected_columns)}"
            )

            logging.info(
                f"Actual number of columns: {len(actual_columns)}"
            )

            missing_columns = set(expected_columns) - set(actual_columns)
            extra_columns = set(actual_columns) - set(expected_columns)

            if missing_columns:
                logging.error(
                    f"Missing columns: {sorted(missing_columns)}"
                )

            if extra_columns:
                logging.error(
                    f"Unexpected columns: {sorted(extra_columns)}"
                )

            if not missing_columns and not extra_columns:
                logging.info("Column validation successful.")
                return True

            return False

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> bool:
        """
        Detect distribution drift using the two-sample KS test.

        Only numerical columns are tested. Raw URL strings are skipped.
        The report records each tested column's p-value and drift status.
        """
        try:
            status = True
            report = {}

            # Use the numerical columns defined in schema.yaml.
            numerical_columns = self._schema_config.get(
                "numerical_columns", []
            )

            for column in numerical_columns:

                if column not in base_df.columns:
                    raise ValueError(
                        f"Column '{column}' is missing from training data."
                    )

                if column not in current_df.columns:
                    raise ValueError(
                        f"Column '{column}' is missing from testing data."
                    )

                # Convert values to numeric; invalid values become NaN.
                base_values = pd.to_numeric(
                    base_df[column], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).dropna()

                current_values = pd.to_numeric(
                    current_df[column], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).dropna()

                # KS test requires usable observations in both samples.
                if len(base_values) == 0 or len(current_values) == 0:
                    logging.warning(
                        f"Skipping '{column}': "
                        "no valid numerical observations in one dataset."
                    )

                    report[column] = {
                        "p_value": None,
                        "drift_status": None,
                        "message": "Insufficient valid numerical data"
                    }

                    # Unable to verify this column's drift status.
                    status = False
                    continue

                # Convert to NumPy arrays for scipy.
                base_array = base_values.to_numpy(dtype=np.float64)
                current_array = current_values.to_numpy(dtype=np.float64)

                # Perform KS test.
                ks_result = ks_2samp(
                    base_array,
                    current_array,
                    nan_policy="omit"
                )

                p_value = float(ks_result.pvalue)
                drift_detected = p_value < threshold

                if drift_detected:
                    status = False

                report[column] = {
                    "p_value": p_value,
                    "drift_status": drift_detected
                }

                logging.info(
                    f"Column: {column} | "
                    f"P-value: {p_value:.6f} | "
                    f"Drift detected: {drift_detected}"
                )

            # Create drift report directory.
            drift_report_file_path = (
                self.data_validation_config.drift_report_file_path
            )

            os.makedirs(
                os.path.dirname(drift_report_file_path),
                exist_ok=True
            )

            # Save drift report.
            write_yaml_file(
                file_path=drift_report_file_path,
                content=report
            )

            logging.info(
                f"Drift report saved: {drift_report_file_path}"
            )

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Validate datasets, detect drift, save validated CSVs,
        and return the validation artifact.
        """
        try:
            # Get ingested dataset paths.
            train_file_path = (
                self.data_ingestion_artifact.trained_file_path
            )

            test_file_path = (
                self.data_ingestion_artifact.test_file_path
            )

            # Read datasets.
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)

            logging.info(
                f"Training dataset shape: {train_dataframe.shape}"
            )

            logging.info(
                f"Testing dataset shape: {test_dataframe.shape}"
            )

            # Validate training columns.
            if not self.validate_number_of_columns(train_dataframe):
                raise ValueError(
                    "Training dataset columns do not match schema.yaml."
                )

            # Validate testing columns.
            if not self.validate_number_of_columns(test_dataframe):
                raise ValueError(
                    "Testing dataset columns do not match schema.yaml."
                )

            logging.info(
                "Training and testing column validation passed."
            )

            # Check whether training and testing distributions differ.
            validation_status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )

            # Paths for validated datasets.
            valid_train_file_path = (
                self.data_validation_config.valid_train_file_path
            )

            valid_test_file_path = (
                self.data_validation_config.valid_test_file_path
            )

            # Create directory for validated datasets.
            os.makedirs(
                os.path.dirname(valid_train_file_path),
                exist_ok=True
            )

            # Save validated datasets.
            train_dataframe.to_csv(
                valid_train_file_path,
                index=False
            )

            test_dataframe.to_csv(
                valid_test_file_path,
                index=False
            )

            logging.info(
                f"Validated training data saved: {valid_train_file_path}"
            )

            logging.info(
                f"Validated testing data saved: {valid_test_file_path}"
            )

            # Create validation artifact.
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=(
                    self.data_validation_config.drift_report_file_path
                )
            )

            logging.info(
                f"Data validation completed: {data_validation_artifact}"
            )

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

