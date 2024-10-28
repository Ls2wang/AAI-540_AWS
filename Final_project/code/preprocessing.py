
"""Feature engineers the Hospital Length of Stay (LOS) dataset."""
import argparse
import logging
import os
import pathlib
import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")

    base_dir = "/opt/ml/processing"

    logger.info("Reading downloaded data.")
    df = pd.read_csv( f"{base_dir}/input/trainData.csv")
    
    # Drop rows missing key information
    df = df.dropna(subset=['Stay'])
    df = df.dropna(subset=['case_id'])
    df = df.dropna(subset=['patientid'])
    
    # Drop 'case_id' column
    df = df.drop(columns=['case_id'])
    
    logger.info("Defining transformers.")
    
    # Impute with mode
    mode_features = [
        "Hospital_code", "City_Code_Hospital", "Age"
    ]
    
    mode_imputer= SimpleImputer(strategy="most_frequent")
    
    df[mode_features] = mode_imputer.fit_transform(df[mode_features])
    
    # Impute with median
    numeric_features = [
        "AvailableExtraRoomsinHospital", "BedGrade", "Admission_Deposit"
    ]
    
    median_imputer = SimpleImputer(strategy="median")

    df[numeric_features] = median_imputer.fit_transform(df[numeric_features])
    
    # Impute with "missing", then ordinal encode
    # Age is already imputed above, so will only be encoded
    categorical_features = [
        "Hospital_type_code", "Hospital_region_code", "Department", "Ward_Type",
        "Ward_Facility_Code", "TypeofAdmission", "SeverityofIllness", "Age"
    ]
    
    miss_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    ord_encoder = OrdinalEncoder()
    
    df[categorical_features] = miss_imputer.fit_transform(df[categorical_features])
    df[categorical_features] = ord_encoder.fit_transform(df[categorical_features])
    df[categorical_features] = df[categorical_features].astype(int)

    # Impute City_Code_Patient value based on City_Code_Hospital
    df['City_Code_Patient'] = df['City_Code_Patient'].fillna(df['City_Code_Hospital'])
    
    # Convert to int
    df['BedGrade'] = df['BedGrade'].astype(int)
    df['AvailableExtraRoomsinHospital'] = df['AvailableExtraRoomsinHospital'].astype(int)
    df['City_Code_Patient'] = df['City_Code_Patient'].astype(int)
    
    # Fill missing VisitorswithPatient with 0
    df['VisitorswithPatient'] = df['VisitorswithPatient'].fillna(0)
    
    # Encode target
    label_encoder = LabelEncoder()
    df['Stay'] = label_encoder.fit_transform(df['Stay'])
    
    df.reset_index(drop=True)
    
    y = df.pop('Stay')
    df.insert(0, 'Stay', y)
    
    print('df head', df.head())
    print('df dtypes', df.dtypes)
    
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(df))
    
    train, split = train_test_split(df, test_size=0.3, random_state=42)
    test, validation = train_test_split(split, test_size=0.5, random_state=42)
    print(train.shape)
    print(test.shape)
    print(validation.shape)

    # Save the datasets into train, validation, and test directories
    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", index=False)

    logger.info("Preprocessing complete.")
