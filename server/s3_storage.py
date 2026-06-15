import os
from pathlib import Path
import boto3

from logger import get_logger

logger = get_logger("S3_Storage")

class S3_Storage:
    # TODO there is no reason why need a class. If the requirements in fact tell
    #      us to include a class, then there is no need to create a static class
    # TODO design and implement an approach to skip S3 connection alltogether
    def UploadFilesToS3(self, source_directory, destination_directory):
        if not os.environ.get('AWS_BUCKET_NAME'):
            print("AWS_BUCKET_NAME environment variable not set. Aborting S3 upload.")
            return

        if not os.environ.get('AWS_ACCESS_KEY_ID'):
            print("AWS_ACCESS_KEY_ID environment variable not set. Aborting S3 upload.")
            return

        bucket_name = os.environ.get('AWS_BUCKET_NAME') 

        print(f"Uploading files from {source_directory} to S3...")
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            aws_session_token = os.environ.get('AWS_SESSION_TOKEN'),
            region_name=os.environ.get('AWS_DEFAULT_REGION'),
            endpoint_url= "https://" + os.environ.get('AWS_S3_ENDPOINT')
        )

        print(f"Using S3 bucket: {bucket_name} and S3 directory: {destination_directory} and local output directory: {source_directory}")

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
                    print(f"Upload {local_path} to s3://{bucket_name}/{s3_key}")
                    s3.upload_file(str(local_path), bucket_name, s3_key)
        except Exception as e:
            print(f"Error uploading files to S3: {e}")


    def DownloadFilesFromS3(source_directory, destination_directory):
        """
        Download all files from the given S3 source_directory (prefix) to the local destination_directory.

        If `AWS_BUCKET_NAME` is defined and non-empty, then we assume that the
        config files should be downloaded from S3.
        """
        bucket_name = os.environ.get('AWS_BUCKET_NAME')
        access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
        secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
        session_token = os.environ.get('AWS_SESSION_TOKEN'),
        default_region = os.environ.get('AWS_DEFAULT_REGION'),
        s3_endpoint = os.environ.get('AWS_S3_ENDPOINT')

        # Mandatory variable
        if bucket_name is None:
            logger.warning("AWS_BUCKET_NAME environment variable not set. Aborting S3 download.")
            return
        if len(bucket_name) <= 0:
            logger.warning("AWS_BUCKET_NAME environment has white-space characters only. Aborting S3 download.")
            return

        # Vault service should for sure have uploaded the below two variables
        # to the env vars.
        #
        # We still need to check for None, as the asserts are removed in
        # production build.
        assert access_key_id, "AWS_ACCESS_KEY_ID was not found in the environment variables, but it should have been."
        assert secret_access_key, "AWS_SECRET_ACCESS_KEY was not found in the environment variables, but it should have been."

        # Mandatory variables
        if not access_key_id:
            logger.error("AWS_ACCESS_KEY_ID environment variable not set. Aborting S3 download.")
            return
        if not secret_access_key:
            logger.error("AWS_SECRET_ACCESS_KEY environment variable not set. Aborting S3 download.")
            return
        
        # Mandatory variable
        if default_region is None:
            logger.error("AWS_DEFAULT_REGION environment variable not set. Aborting S3 download.")
            return
        if len(default_region.strip()) <= 0:
            logger.error("AWS_DEFAULT_REGION environment variable has white-space characters only. Aborting S3 download.")
            return
        
        # Mandatory variable
        if s3_endpoint is None:
            logger.error("AWS_S3_ENDPOINT environment variable not set. Aborting S3 download.")
            return
        if len(s3_endpoint.strip()) <= 0:
            logger.error("AWS_S3_ENDPOINT environment variable has white-space characters only. Aborting S3 download.")
            return


        # Optional variable
        if len(session_token.strip()) <= 0:
            # The session token is optional, but if it is empty, it is a sign that
            # something went wrong.
            assert True, "Environment variable 'AWS_SESSION_TOKEN' is empty. Even though it is optional, it should not white characters only."
            logger.error("Environment variable 'AWS_SESSION_TOKEN' has white characters only. This should not happen. Aborting S3 download")
            session_token = None

        logger.info(f"Downloading files from s3://{bucket_name}/{source_directory} to {destination_directory}...")
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            region_name=default_region,
            endpoint_url="https://" + s3_endpoint
        )

        paginator = s3.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=bucket_name, Prefix=source_directory):
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
                    logger.info(f"Downloading {s3_key} to {local_path}")
                    s3.download_file(bucket_name, s3_key, str(local_path))
        except Exception as e:
            logger.error(f"Error downloading files from S3: {e}")