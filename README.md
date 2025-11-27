# GCP Tech Influencer Agent

A serverless, autonomous agent that posts trending tech content to X (Twitter). Built with Google Cloud Run Jobs, Vertex AI (Gemini + Veo), and Firestore.

## Architecture

- **Orchestrator**: Cloud Run Jobs (Python container)
- **Brain**: Vertex AI Gemini 1.5 Flash (Strategy & Scripting)
- **Video**: Vertex AI Veo (Video Generation)
- **Database**: Firestore (History & Deduplication)
- **Secrets**: Secret Manager (API Keys)
- **Scheduler**: Cloud Scheduler (7 daily triggers, AWST timezone)
- **IaC**: Terraform

## Prerequisites

- Google Cloud Project with Billing enabled
- X (Twitter) Developer Account with v2 API access
- `gcloud` CLI installed and authenticated
- Terraform installed (v1.0+)
- Firestore database already created in your project

## Quick Start (Terraform)

```bash
# 1. Clone and navigate to terraform directory
cd terraform

# 2. Run the setup script with your project ID
./setup.sh YOUR_PROJECT_ID
```

The setup script will:
1. Create APIs, Artifact Registry, and Service Account
2. Build and push the Docker container
3. Create Cloud Run Job and Cloud Scheduler triggers

## Manual Deployment

If you prefer manual steps:

```bash
# Set your project
export PROJECT_ID="your-project-id"
export REGION="us-central1"

# Initialize and apply Terraform (prerequisites first)
cd terraform
terraform init
terraform apply -target=google_project_service.required_apis \
    -target=google_artifact_registry_repository.phantom_repo \
    -target=google_service_account.phantom_sa

# Build container image
cd ..
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/phantom-influencer/phantom-influencer:latest .

# Apply remaining Terraform (Cloud Run Job, Scheduler)
cd terraform
terraform apply
```

## Configure Twitter Secrets

After deployment, add your Twitter API credentials:

```bash
echo -n 'your-consumer-key' | gcloud secrets versions add TWITTER_CONSUMER_KEY --data-file=-
echo -n 'your-consumer-secret' | gcloud secrets versions add TWITTER_CONSUMER_SECRET --data-file=-
echo -n 'your-access-token' | gcloud secrets versions add TWITTER_ACCESS_TOKEN --data-file=-
echo -n 'your-access-token-secret' | gcloud secrets versions add TWITTER_ACCESS_TOKEN_SECRET --data-file=-
echo -n 'your-bearer-token' | gcloud secrets versions add TWITTER_BEARER_TOKEN --data-file=-  # Optional
```

## Rebuild After Code Changes

After pushing new code to GitHub and pulling in GCP Cloud Shell:

```bash
cd ~/phantom
git pull

# Build new image and update Cloud Run job
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/phantom-influencer/phantom-influencer:latest .
gcloud run jobs update phantom-influencer-job --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/phantom-influencer/phantom-influencer:latest --region ${REGION}
```

## Useful Commands

```bash
# Test the job (force post)
gcloud run jobs execute phantom-influencer-job --region us-central1 --update-env-vars FORCE_POST=true

# View logs
gcloud logging read 'resource.type=cloud_run_job AND resource.labels.job_name=phantom-influencer-job' --limit=50

# View scheduler status
gcloud scheduler jobs list --location=us-central1

# Check Terraform outputs
cd terraform && terraform output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ID` | Your GCP Project ID | Required |
| `REGION` | GCP region for resources | `us-central1` |
| `BUDGET_MODE` | `True` to disable video generation | `True` |
| `TIMEZONE` | Scheduler timezone | `Australia/Perth` |
| `CURRENCY` | Currency for budget display | `AUD` |

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires GCP credentials)
python main.py
```

## License

MIT
