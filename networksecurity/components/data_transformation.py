import sys
import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
    SCHEMA_FILE_PATH
)

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
    read_yaml_file
)


class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Initializing KNNImputer")

            imputer = KNNImputer(
                **DATA_TRANSFORMATION_IMPUTER_PARAMS
            )

            processor = Pipeline(
                steps=[
                    ("imputer", imputer)
                ]
            )

            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")

            # Read validated datasets
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )

            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Testing data shape: {test_df.shape}")

            # Validate target column
            if TARGET_COLUMN not in train_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' is missing from training data"
                )

            if TARGET_COLUMN not in test_df.columns:
                raise ValueError(
                    f"Target column '{TARGET_COLUMN}' is missing from testing data"
                )

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Get numerical columns from schema
            # Exclude the target column even if it appears in the schema list
            numerical_columns = [
                column
                for column in self._schema_config.get(
                    "numerical_columns", []
                )
                if column != TARGET_COLUMN
            ]

            if not numerical_columns:
                raise ValueError(
                    "No numerical input features found in schema.yaml"
                )

            # Validate numerical columns
            missing_train_columns = (
                set(numerical_columns)
                - set(input_feature_train_df.columns)
            )

            missing_test_columns = (
                set(numerical_columns)
                - set(input_feature_test_df.columns)
            )

            if missing_train_columns:
                raise ValueError(
                    "Missing numerical training columns: "
                    f"{sorted(missing_train_columns)}"
                )

            if missing_test_columns:
                raise ValueError(
                    "Missing numerical testing columns: "
                    f"{sorted(missing_test_columns)}"
                )

            # Select numerical features only.
            # Raw URL strings are excluded from KNNImputer.
            input_feature_train_df = input_feature_train_df[
                numerical_columns
            ].copy()

            input_feature_test_df = input_feature_test_df[
                numerical_columns
            ].copy()

            # Convert feature values to numeric
            input_feature_train_df = input_feature_train_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            input_feature_test_df = input_feature_test_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # Replace infinite values with NaN
            input_feature_train_df = input_feature_train_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            input_feature_test_df = input_feature_test_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # Check columns containing no valid training values
            all_missing_columns = (
                input_feature_train_df.columns[
                    input_feature_train_df.isna().all()
                ].tolist()
            )

            if all_missing_columns:
                raise ValueError(
                    "These training features contain no valid values: "
                    f"{all_missing_columns}"
                )

            # Initialize preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit only on training data
            preprocessor_object = preprocessor.fit(
                input_feature_train_df
            )

            # Transform training and testing data
            transformed_input_train_feature = (
                preprocessor_object.transform(
                    input_feature_train_df
                )
            )

            transformed_input_test_feature = (
                preprocessor_object.transform(
                    input_feature_test_df
                )
            )

            # Check transformed values
            if not np.isfinite(
                transformed_input_train_feature
            ).all():
                raise ValueError(
                    "Invalid values remain in transformed training data"
                )

            if not np.isfinite(
                transformed_input_test_feature
            ).all():
                raise ValueError(
                    "Invalid values remain in transformed testing data"
                )

            # Combine features and target
            train_arr = np.c_[
                transformed_input_train_feature,
                target_feature_train_df.to_numpy()
            ]

            test_arr = np.c_[
                transformed_input_test_feature,
                target_feature_test_df.to_numpy()
            ]

            # Create output directory
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_train_file_path
                ),
                exist_ok=True
            )

            # Save transformed NumPy arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # Save fitted preprocessing object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
            )

            logging.info(
                f"Transformed training data shape: {train_arr.shape}"
            )

            logging.info(
                f"Transformed testing data shape: {test_arr.shape}"
            )

            # Create transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=(
                    self.data_transformation_config.transformed_object_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config.transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config.transformed_test_file_path
                )
            )

            logging.info(
                "Data transformation completed successfully"
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)