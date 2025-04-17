import os
import boto3
import pandas as pd
import io
from dotenv import load_dotenv
from pandas.errors import EmptyDataError

# Load environment variables from .env file
load_dotenv()

# Configuration dictionary
config = {
    "access_key_id": os.getenv("Access_key_ID"),
    "secret_access_key": os.getenv("Secret_access_key"),
    "bucket_name": os.getenv("BUCKET_NAME"),
    "local_path": os.getenv("local_path"),
    "region_name": os.getenv("REGION_NAME"),
}

# Initialize Boto3 S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=config["access_key_id"],
    aws_secret_access_key=config["secret_access_key"],
    region_name=config["region_name"],
)

bucket_name = config["bucket_name"]
s3_base_prefix = "raw-data/"
local_folder = config["local_path"]

# Process each file in the local folder
for filename in os.listdir(local_folder):
    file_path = os.path.join(local_folder, filename)

    # Handle Excel files
    if filename.endswith(".xlsx"):
        try:
            workbook_name = os.path.splitext(filename)[0]
            print(f"Processing workbook: {filename}")
            xls = pd.ExcelFile(file_path)

            for sheet in xls.sheet_names:
                print(f"  - Sheet: {sheet}")
                try:
                    df = pd.read_excel(xls, sheet_name=sheet).dropna(how="all")

                    # Clean sheet name for S3 key
                    clean_sheet_name = sheet.lower().replace(" ", "_")
                    s3_key = f"{s3_base_prefix}{workbook_name}/{clean_sheet_name}.csv"

                    # Save to S3
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    s3.put_object(
                        Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue()
                    )
                    print(f"Uploaded to s3://{bucket_name}/{s3_key}")

                except (ValueError, pd.errors.ParserError) as sheet_err:
                    print(f"Error reading sheet '{sheet}' in '{filename}': {sheet_err}")

        except (FileNotFoundError, OSError, PermissionError) as file_err:
            print(f"Failed to process Excel file '{filename}': {file_err}")

    # Handle CSV files
    elif filename.endswith(".csv"):
        try:
            base_name = os.path.splitext(filename)[0]
            print(f"Processing single CSV: {filename}")
            df = pd.read_csv(file_path).dropna(how="all")

            s3_key = f"{s3_base_prefix}{base_name}/{base_name}.csv"

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
            print(f"Uploaded to s3://{bucket_name}/{s3_key}")

        except (
            FileNotFoundError,
            PermissionError,
            EmptyDataError,
            pd.errors.ParserError,
        ) as csv_err:
            print(f"Failed to process CSV file '{filename}': {csv_err}")
