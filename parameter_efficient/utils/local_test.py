import os
import boto3
import tarfile

# Set up S3 client
s3 = boto3.client("s3")


def download_and_extract_model(s3_bucket, s3_key, model_dir):
    """
    Downloads and extracts a model from an S3 bucket.

    Args:
    - s3_bucket (str): S3 bucket name.
    - s3_key (str): S3 object key (path to the model tar.gz file).
    - model_dir (str): Local directory where the model should be saved and extracted.
    """

    model_path = os.path.join(model_dir, "model.pth")

    if os.path.exists(model_path):
        print("Model already exists")
        return

    # Local path to save the model tar.gz file
    local_tar_path = os.path.join(model_dir, "model.tar.gz")

    # Download model from S3
    s3.download_file(s3_bucket, s3_key, local_tar_path)
    print(f"Downloaded {s3_key} from S3 bucket {s3_bucket}")

    # Extract the tar.gz file
    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    print(f"Extracted model to {model_dir}")

    # Clean up the tar.gz file after extraction
    os.remove(local_tar_path)
    print("Removed the tar.gz file")
