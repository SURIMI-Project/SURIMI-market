import os
from pathlib import Path
import boto3

from .vault_service import VaultService

# TODO design and implement an approach to skip S3 connection alltogether
def upload_files_to_S3(vault_service: VaultService, source_directory, destination_directory):
    if len(vault_service.aws_bucket_name) == 0:
        print("AWS_BUCKET_NAME from Vault is not set. Aborting S3 upload.")
        return

    if len(vault_service.aws_acccess_key_id) == 0:
        print("AWS_ACCESS_KEY_ID from vault_service is not set. Aborting S3 upload.")
        return

    print(f"Uploading files from {source_directory} to S3...")
    s3 = boto3.client(
        's3',
        aws_access_key_id=vault_service.aws_acccess_key_id,
        aws_secret_access_key=vault_service.aws_secret_access_key,
        aws_session_token = vault_service.aws_session_token,
        region_name=vault_service.default_region,
        endpoint_url= "https://" + vault_service.aws_s3_endpoint
    )

    print(f"Using S3 bucket: {vault_service.aws_bucket_name} and S3 directory: {destination_directory} and local output directory: {source_directory}")

    try:
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                if file.endswith(".bin"):
                    continue  # Skip .bin files
                print(f"Uploading {file} from {root} to S3...")
                local_path = Path(root) / file
                # Compute the S3 key by replacing the local_sim_dir part with s3_sim_dir
                relative_path = local_path.relative_to(source_directory)
                s3_key = f"{destination_directory}/{relative_path.as_posix()}"
                print(f"Upload {local_path} to s3://{vault_service.aws_bucket_name}/{s3_key}")
                s3.upload_file(str(local_path), vault_service.aws_bucket_name, s3_key)
    except Exception as e:
        print(f"Error uploading files to S3: {e}")


def download_files_from_S3(vault_service: VaultService, source_directory, destination_directory):
    """
    Download all files from the given S3 source_directory (prefix) to the local destination_directory.
    """
    if len(vault_service.aws_bucket_name) == 0:
        print("AWS_BUCKET_NAME from Vault is not set. Aborting S3 upload.")
        return

    print(f"Downloading files from s3://{vault_service.aws_bucket_name}/{source_directory} to {destination_directory}...")
    s3 = boto3.client(
        's3',
        aws_access_key_id=vault_service.aws_acccess_key_id,
        aws_secret_access_key=vault_service.aws_secret_access_key,
        aws_session_token = vault_service.aws_session_token,
        region_name=vault_service.default_region,
        endpoint_url= "https://" + vault_service.aws_s3_endpoint
    )

    paginator = s3.get_paginator('list_objects_v2')
    try:
        for page in paginator.paginate(Bucket=vault_service.aws_bucket_name, Prefix=source_directory):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                # Skip directories
                if s3_key.endswith('/'):
                    continue
                # Skip .keep files. They are used to keep empty directories in S3. (directories may not be empty in S3)
                if s3_key.endswith('.keep'):
                    continue                    
                # Compute local file path
                relative_path = Path(s3_key).relative_to(source_directory)
                local_path = Path(destination_directory) / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading {s3_key} to {local_path}")
                s3.download_file(vault_service.aws_bucket_name, s3_key, str(local_path))
    except Exception as e:
        print(f"Error downloading files from S3: {e}")