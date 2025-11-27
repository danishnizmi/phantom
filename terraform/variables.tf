# Phantom Tech Influencer - Terraform Variables
# Configure these for your GCP project

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for Cloud Run and related services"
  type        = string
  default     = "us-central1"
}

variable "timezone" {
  description = "Timezone for scheduling (AWST = Australia/Perth)"
  type        = string
  default     = "Australia/Perth"
}

variable "budget_mode" {
  description = "Enable budget mode to disable video generation"
  type        = bool
  default     = false
}

variable "image_name" {
  description = "Name for the container image"
  type        = string
  default     = "phantom-influencer"
}

variable "job_name" {
  description = "Name for the Cloud Run Job"
  type        = string
  default     = "phantom-influencer-job"
}

variable "service_account_email" {
  description = "Service account email for the Cloud Run Job (leave empty to create new)"
  type        = string
  default     = ""
}

# Scheduler configuration
variable "scheduler_triggers" {
  description = "Cron expressions for Cloud Scheduler triggers (AWST timezone)"
  type        = list(string)
  default = [
    "30 7 * * *",   # 7:30 AM AWST - Morning
    "15 10 * * *",  # 10:15 AM AWST - Late morning
    "45 12 * * *",  # 12:45 PM AWST - Lunch
    "30 15 * * *",  # 3:30 PM AWST - Afternoon
    "0 18 * * *",   # 6:00 PM AWST - Early evening
    "30 20 * * *",  # 8:30 PM AWST - Peak evening
    "15 22 * * *",  # 10:15 PM AWST - Night
  ]
}

# Resource limits
variable "cpu_limit" {
  description = "CPU limit for Cloud Run Job (e.g., '1', '2')"
  type        = string
  default     = "1"
}

variable "memory_limit" {
  description = "Memory limit for Cloud Run Job (e.g., '512Mi', '1Gi', '2Gi')"
  type        = string
  default     = "1Gi"
}

variable "timeout_seconds" {
  description = "Timeout for Cloud Run Job execution"
  type        = number
  default     = 900  # 15 minutes
}

variable "max_retries" {
  description = "Maximum retries for failed job executions"
  type        = number
  default     = 1
}

# Labels
variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    app         = "phantom-influencer"
    environment = "production"
    managed-by  = "terraform"
  }
}
