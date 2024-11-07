"""This file contains code for parsing command line arguments."""

import os
import argparse


def parse_flags() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PEFT model using SageMaker.")

    parser.add_argument("--type", type=str, help="Task type")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,  # QQP is a binary classification task
        help="Number of labels for classification",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--logging-steps", type=int, default=50, help="Log every X updates steps"
    )
    parser.add_argument(
        "--tokenizer-model-name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Tokenizer model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Model to train name",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=4,  # Balance between parameter efficiency and the model's ability to learn.
        help="""
            Rank of the LoRA adaptation matrix
            Controls the number of additional parameters
        """,
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,  # Sufficient learning while avoiding overfitting,
        help="""
            Scaling factor (alpha) for the LoRA adaptation matrix
            Controls the strength of the adaptation.
        """,
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1, help="Dropout rate for LoRA layers"
    )
    parser.add_argument(
        "--lora-bias",
        type=str,
        default="none",
        help="Bias type for LoRA layers. Set to 'none' for no bias.",
    )
    parser.add_argument(
        "--lora-task-type",
        type=str,
        default="SEQ_CLS",  # Sequence Classification
        help="Task type for LoRA configuration",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,  # no bias in the LoRA layers
        help="""
            S3 Directory containing the dataset.
            If not specified, the dataset will be defaulted to QQP data from Hugging Face.
        """,
    )
    parser.add_argument(
        "--use-spot-instances",
        type=bool,
        default=False,
        help="Whether to use spot instances or not",
    )
    parser.add_argument(
        "--instance-count", type=int, default=1, help="Number of EC2 instances to use."
    )
    parser.add_argument(
        "--instance-type", type=str, default="ml.p3.2xlarge", help="EC2 instance type."
    )
    parser.add_argument(
        "--evaluate-instance-type",
        type=str,
        default="ml.g4dn.xlarge",
        help="EC2 instance type for evaluation.",
    )
    parser.add_argument(
        "--sagemaker-pytorch-training-version",
        type=str,
        default="pytorch-training-2024-09-19-09-17-17-352",
        help="""
            PyTorch Training version on Sagemaker training jobs.
            Eg: pytorch-training-2024-09-19-05-07-56-118
        """,
    )
    parser.add_argument(
        "--sagemaker-bucket",
        type=str,
        default="peft-learning",
        help="S3 bucket for Sagemaker",
    )

    return parser.parse_args()


def get_hyperparameters():
    epochs = int(os.getenv("EPOCHS", "1"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    learning_rate = float(os.getenv("LEARNING_RATE", "1e-3"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "0.01"))
    logging_steps = int(os.getenv("LOGGING_STEPS", "50"))
    num_labels = int(os.getenv("NUM_LABELS", "2"))
    tokenizer_model_name = os.getenv(
        "TOKENIZER_MODEL_NAME", "microsoft/deberta-v3-base"
    )
    model_name = os.getenv("MODEL_NAME", "microsoft/deberta-v3-base")
    lora_r = int(os.getenv("LORA_R", "4"))
    lora_alpha = int(os.getenv("LORA_ALPHA", "16"))
    lora_dropout = float(os.getenv("LORA_DROPOUT", "0.1"))
    lora_task_type = os.getenv("LORA_TASK_TYPE", "SEQ_CLS")
    lora_bias = os.getenv("LORA_BIAS", "none")
    data_dir = os.getenv("DATA_DIR", None)
    use_spot_instances = bool(os.getenv("USE_SPOT_INSTANCES", "False") == "True")
    instance_count = int(os.getenv("INSTANCE_COUNT", "1"))
    instance_type = os.getenv("INSTANCE_TYPE", "ml.p3.2xlarge")
    evaluate_instance_type = os.getenv("EVALUATE_INSTANCE_TYPE", "ml.g4dn.xlarge")
    sagemaker_pytorch_training_version = os.getenv(
        "SAGEMAKER_PYTORCH_TRAINING_VERSION", "pytorch-training-2024-09-19-09-17-17-352"
    )
    sagemaker_bucket = os.getenv("SAGEMAKER_BUCKET", "peft-learning")

    return argparse.Namespace(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        num_labels=num_labels,
        tokenizer_model_name=tokenizer_model_name,
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_task_type=lora_task_type,
        lora_bias=lora_bias,
        data_dir=data_dir,
        use_spot_instances=use_spot_instances,
        instance_count=instance_count,
        instance_type=instance_type,
        evaluate_instance_type=evaluate_instance_type,
        sagemaker_pytorch_training_version=sagemaker_pytorch_training_version,
        sagemaker_bucket=sagemaker_bucket,
    )
