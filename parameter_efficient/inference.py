import os
import csv
import json
import torch
from transformers import AutoTokenizer
import boto3
import tarfile
import tempfile

try:
    # Local environment (when parameter_efficient is a package)
    from parameter_efficient.utils import get_device, get_hyperparameters
    from parameter_efficient.model import create_peft_model
except ImportError:
    # SageMaker environment (no package structure, just files in source_dir)
    from utils import get_device, get_hyperparameters
    from model import create_peft_model


# Load model and tokenizer
# pylint: disable=unused-argument
def model_fn(model_dir):

    args = get_hyperparameters()

    device = get_device()
    # Initialize S3 client
    s3 = boto3.client("s3")

    s3_bucket = args.sagemaker_bucket
    s3_key = f"output/{args.sagemaker_pytorch_training_version}/output/model.tar.gz"

    # Step 1: Download the .tar.gz file from S3
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory to extract the model
    tar_file_path = os.path.join(temp_dir, "model.tar.gz")

    try:
        # Download model.tar.gz from S3 to local temp_dir
        with open(tar_file_path, "wb") as f:
            s3.download_fileobj(s3_bucket, s3_key, f)

        # Step 2: Extract the tar.gz file
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Ensure model.pth exists in the extracted content
        model_path = os.path.join(temp_dir, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model.pth not found in extracted archive: {model_path}"
            )

        # Step 3: Load the model.pth into the DeBERTa base model
        model = create_peft_model(args)

        # Get the device
        device = get_device()

        # Load the saved state_dict from model.pth
        model_state_dict = torch.load(
            model_path, map_location=torch.device(device), weights_only=False
        )
        model.load_state_dict(model_state_dict)

        # Step 4: Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model_name)

    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {e}") from e

    return model, tokenizer


# Preprocess input
def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        # Parse the CSV input (assuming the format has 'question1' and 'question2' columns)
        reader = csv.DictReader(request_body.splitlines())
        questions1, questions2 = [], []
        for row in reader:
            questions1.append(row["question1"])
            questions2.append(row["question2"])
        return questions1, questions2
    raise ValueError(f"Unsupported content type: {request_content_type}")


# Make predictions
def predict_fn(input_data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    device = get_device()
    model.to(device)
    questions1, questions2 = input_data
    inputs = tokenizer(
        questions1,
        questions2,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
    return predictions


# Format the output
def output_fn(prediction):
    return json.dumps({"predictions": prediction})
