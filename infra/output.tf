output "peft_bucket" {
  value = module.s3.aws_s3_bucket_peft_bucket
}

output "peft_sagemaker_execution_role" {
  value = module.iam.aws_iam_role_peft_sagemaker_execution_role_arn
}
