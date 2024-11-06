terraform {
  backend "s3" {
    bucket = "peft-terraform-state"
    key = "terraform.tfstate"
    region = "us-east-1"
  }
}

module "s3" {
  source = "./modules/s3"
}

module "iam" {
  source = "./modules/iam"
  aws_s3_bucket_peft_arn = module.s3.aws_s3_bucket_peft_arn
}
