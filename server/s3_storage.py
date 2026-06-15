import os
from pathlib import Path
import boto3

class S3_Storage:
    # TODO there is no reason why need a class. If the requirements in fact tell
    #      us to include a class, then there is no need to create a static class
    def DownloadFilesFromS3(source_directory, destination_directory):
        """
        Download all files from the given S3 source_directory (prefix) to the local destination_directory.
        """
        if not os.environ.get('AWS_BUCKET_NAME'):
            print("AWS_BUCKET_NAME environment variable not set. Aborting S3 download.")
            return

        bucket_name = os.environ.get('AWS_BUCKET_NAME')

        print(f"Downloading files from s3://{bucket_name}/{source_directory} to {destination_directory}...")
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.environ.get('AWS_SESSION_TOKEN'),
            region_name=os.environ.get('AWS_DEFAULT_REGION'),
            endpoint_url="https://" + os.environ.get('AWS_S3_ENDPOINT')
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
                    print(f"Downloading {s3_key} to {local_path}")
                    s3.download_file(bucket_name, s3_key, str(local_path))
        except Exception as e:
            print(f"Error downloading files from S3: {e}")