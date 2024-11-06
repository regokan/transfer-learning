# AWS Access Key
variable "aws_access_key" {
  description = "AWS Access Key"
  type        = string
}

# AWS Secret Key
variable "aws_secret_key" {
  description = "AWS Secret Key"
  type        = string
}

# AWS Region
variable "aws_region" {
  description = "AWS Region"
  default     = "us-east-1"
  type        = string
}
