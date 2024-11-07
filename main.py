import os
from sys import exit
from dotenv import load_dotenv
from deploy import (
    sagemaker_train_peft,
    sagemaker_evaluate_peft,
    sagemaker_evaluate_local_peft,
)
from parameter_efficient.utils import parse_flags

load_dotenv()

sagemaker_bucket = os.getenv("PEFT_BUKET")
sagemaker_role = os.getenv("PEFT_SAGEMAKER_EXECUTION_ROLE")

if sagemaker_bucket is None or sagemaker_role is None:
    print(
        "Please set the environment variables PEFT_BUKET and PEFT_SAGEMAKER_EXECUTION_ROLE"
    )
    exit(1)


def main():
    args = parse_flags()

    if args.type == "train":
        sagemaker_train_peft(sagemaker_role, sagemaker_bucket, args)
    elif args.type == "evaluate":
        sagemaker_evaluate_peft(sagemaker_role, sagemaker_bucket, args)
    elif args.type == "evaluate-local":
        sagemaker_evaluate_local_peft(sagemaker_bucket, args)
    else:
        print(
            "Please provide a valid type of execution: train, evaluate, evaluate-local"
        )
        exit(1)


if __name__ == "__main__":
    main()
