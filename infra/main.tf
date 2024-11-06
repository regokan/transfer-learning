terraform {
  backend "s3" {
    bucket = "peft-terraform-state"
    key = "terraform.tfstate"
    region = "us-east-1"
  }
}
