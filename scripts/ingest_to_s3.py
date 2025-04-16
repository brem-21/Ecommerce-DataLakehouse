import os
import sys
from typing import Dict, Any
import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


def load_environment_variables() -> Dict[str, Any]:
    """Load and validate required environment variables."""
    load_dotenv()

    config = {
        "access_key_id": os.getenv("Access_key_ID"),
        "secret_access_key": os.getenv("Secret_access_key"),
        "bucket_name": os.getenv("BUCKET_NAME"),
        "local_path": os.getenv("local_path"),
        "region_name": os.getenv("REGION_NAME"),
    }

    missing_keys = [key for key, value in config.items() if not value]
    if missing_keys:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing_keys)}"
        )

    return config


def initialize_s3_client(access_key: str, secret_key: str, region: str) -> boto3.client:
    """Initialize and return an S3 client."""
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to initialize S3 client: {str(e)}") from e


def convert_xlsx_to_csv(local_path: str) -> None:
    """Convert all .xlsx files in the directory to .csv."""
    files = os.listdir(local_path)
    for file in files:
        if file.endswith(".xlsx"):
            file_path = os.path.join(local_path, file)
            try:
                df = pd.read_excel(file_path)
                csv_file = file.replace(".xlsx", ".csv")
                csv_path = os.path.join(local_path, csv_file)
                df.to_csv(csv_path, index=False)
                print(f" Converted: {file} → {csv_file}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f" Failed to convert {file}: {str(e)}", file=sys.stderr)


def upload_files_to_s3(
    s3_client: boto3.client, bucket_name: str, local_path: str, s3_prefix: str
) -> None:
    """Upload all .csv files from local_path to s3_prefix in the bucket."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    # Convert .xlsx files to .csv
    convert_xlsx_to_csv(local_path)

    # Upload only .csv files
    files = [f for f in os.listdir(local_path) if f.endswith(".csv")]
    if not files:
        print(f" No .csv files found in {local_path}")
        return

    for file in files:
        file_path = os.path.join(local_path, file)
        if os.path.isfile(file_path):
            s3_key = f"{s3_prefix}/{file}"

            try:
                s3_client.upload_file(file_path, bucket_name, s3_key)
                print(f" Uploaded: {file_path} → s3://{bucket_name}/{s3_key}")
            except (BotoCoreError, ClientError, IOError) as e:
                print(f" Error uploading {file_path}: {str(e)}", file=sys.stderr)


def main() -> None:
    """Main function to orchestrate the S3 upload process."""
    try:
        # Load configuration
        config = load_environment_variables()

        # Initialize S3 client
        s3_client = initialize_s3_client(
            config["access_key_id"], config["secret_access_key"], config["region_name"]
        )

        # Upload all .csv files in local_path to the S3 folder raw_data/
        upload_files_to_s3(
            s3_client, config["bucket_name"], config["local_path"], "raw_data"
        )

        print("All upload operations completed.")

    except (EnvironmentError, RuntimeError, FileNotFoundError) as e:
        print(f" Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f" Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
