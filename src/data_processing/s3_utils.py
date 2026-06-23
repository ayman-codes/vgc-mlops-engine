import boto3
import os
from botocore.exceptions import ClientError

def upload_file_to_s3(local_path: str, bucket_name: str, s3_key: str) -> bool:
    """
    Uploads a local file to an S3 bucket.
    
    Args:
        local_path: The path to the file to upload.
        bucket_name: The target S3 bucket.
        s3_key: The destination path inside the S3 bucket.
        
    Returns:
        True if successful, False otherwise.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
    )
    
    try:
        s3_client.upload_file(local_path, bucket_name, s3_key)
        print(f"Success: Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"S3 Upload Error: {e}")
        return False