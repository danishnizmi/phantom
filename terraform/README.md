# Terraform Infrastructure for Phantom Tech Influencer

This directory contains Terraform configuration for deploying the Phantom Tech Influencer to Google Cloud Platform.

## Prerequisites

1. **Terraform** >= 1.0.0
   ```bash
   # Install on macOS
   brew install terraform

   # Install on Linux
   curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
   sudo apt update && sudo apt install terraform
   ```

2. **Google Cloud SDK** (gcloud CLI)
   ```bash
   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   ```

3. **GCP Project** with billing enabled

## Quick Start

### 1. Configure Variables

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID and settings
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Preview Changes

```bash
terraform plan
```

### 4. Apply Infrastructure

```bash
terraform apply
```

### 5. Configure Secrets

After Terraform creates the secret placeholders, add your Twitter API credentials:

```bash
# Set your Twitter API credentials
echo -n "your_consumer_key" | gcloud secrets versions add TWITTER_CONSUMER_KEY --data-file=-
echo -n "your_consumer_secret" | gcloud secrets versions add TWITTER_CONSUMER_SECRET --data-file=-
echo -n "your_access_token" | gcloud secrets versions add TWITTER_ACCESS_TOKEN --data-file=-
echo -n "your_access_token_secret" | gcloud secrets versions add TWITTER_ACCESS_TOKEN_SECRET --data-file=-

# Optional: Bearer token for advanced features
echo -n "your_bearer_token" | gcloud secrets versions add TWITTER_BEARER_TOKEN --data-file=-
```

### 6. Build and Push Container

```bash
cd ..
./deploy.sh
```

### 7. Test the Job

```bash
gcloud run jobs execute phantom-influencer-job --region=us-central1
```

## Resources Created

Terraform will create the following GCP resources:

| Resource | Description |
|----------|-------------|
| **Cloud Run Job** | The main application job |
| **Cloud Scheduler** | 7 scheduled triggers (AWST timezone) |
| **Artifact Registry** | Docker container repository |
| **Firestore Database** | Post history and state storage |
| **Service Account** | IAM identity for the job |
| **Secret Manager Secrets** | Twitter API credential storage |

## Cost Estimates (AUD)

Approximate monthly costs for typical usage:

| Service | Cost |
|---------|------|
| Cloud Run | ~$5-15/month (depends on posting frequency) |
| Cloud Scheduler | Free (< 3 jobs free tier) |
| Firestore | ~$0-5/month (depends on history size) |
| Secret Manager | ~$0.50/month |
| Artifact Registry | ~$0.50/month |
| Vertex AI (Gemini) | ~$5-20/month (API calls) |
| Vertex AI (Imagen) | ~$10-30/month (image generation) |
| Vertex AI (Veo) | ~$15-50/month (video generation, if enabled) |

**Total estimated: ~$35-120 AUD/month**

Enable `budget_mode = true` to disable video generation and reduce costs.

## Terraform State

For production, configure remote state storage:

```hcl
# In main.tf, uncomment and configure:
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "phantom-influencer"
  }
}
```

Create the state bucket:
```bash
gsutil mb -l us-central1 gs://your-terraform-state-bucket
gsutil versioning set on gs://your-terraform-state-bucket
```

## Common Operations

### Update Scheduler Times

Edit `terraform.tfvars`:
```hcl
scheduler_triggers = [
  "0 8 * * *",   # 8:00 AM
  "0 12 * * *",  # 12:00 PM
  "0 18 * * *",  # 6:00 PM
]
```

Then apply:
```bash
terraform apply
```

### Enable Budget Mode

```hcl
budget_mode = true
```

### Destroy All Resources

```bash
terraform destroy
```

**Warning**: This will delete all infrastructure including Firestore data!

## Troubleshooting

### API Not Enabled

If you see "API not enabled" errors:
```bash
terraform apply -target=google_project_service.required_apis
```

### Permission Denied

Ensure you have Owner or Editor role on the GCP project:
```bash
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### Firestore Already Exists

Firestore can only be created once per project. If it already exists, import it:
```bash
terraform import google_firestore_database.phantom_db "(default)"
```

## Files

| File | Purpose |
|------|---------|
| `main.tf` | Main resource definitions |
| `variables.tf` | Variable declarations |
| `outputs.tf` | Output values |
| `terraform.tfvars.example` | Example configuration |
| `terraform.tfvars` | Your configuration (git-ignored) |
