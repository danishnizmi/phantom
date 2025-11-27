# Terraform Infrastructure for Phantom Tech Influencer

Infrastructure as Code for deploying the Phantom Tech Influencer to GCP.

## Quick Start (One Command)

```bash
cd terraform
./setup.sh YOUR_PROJECT_ID
```

This will:
1. Create all GCP infrastructure (APIs, IAM, Firestore, etc.)
2. Build and push the container image
3. Create Cloud Run Job with 7 scheduled triggers (AWST timezone)
4. Output commands to configure Twitter API secrets

## Manual Setup

### 1. Set Project ID

```bash
# Option A: Environment variable
export TF_VAR_project_id="your-project-id"

# Option B: Create tfvars file
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
```

### 2. Initialize & Apply

```bash
terraform init
terraform plan
terraform apply
```

### 3. Build Container

```bash
cd ..
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT/phantom-influencer/phantom-influencer .
```

### 4. Configure Secrets

```bash
echo -n 'YOUR_KEY' | gcloud secrets versions add TWITTER_CONSUMER_KEY --data-file=-
echo -n 'YOUR_SECRET' | gcloud secrets versions add TWITTER_CONSUMER_SECRET --data-file=-
echo -n 'YOUR_TOKEN' | gcloud secrets versions add TWITTER_ACCESS_TOKEN --data-file=-
echo -n 'YOUR_TOKEN_SECRET' | gcloud secrets versions add TWITTER_ACCESS_TOKEN_SECRET --data-file=-
```

### 5. Test

```bash
gcloud run jobs execute phantom-influencer-job --region us-central1 --update-env-vars FORCE_POST=true
```

## Resources Created

| Resource | Description |
|----------|-------------|
| Cloud Run Job | Main application |
| Cloud Scheduler (x7) | Triggers throughout the day (AWST) |
| Artifact Registry | Container repository |
| Firestore Database | Post history storage |
| Service Account | IAM identity with required roles |
| Secret Manager Secrets | Twitter API credential storage |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `project_id` | (required) | GCP Project ID |
| `region` | `us-central1` | GCP Region |
| `timezone` | `Australia/Perth` | AWST timezone |
| `budget_mode` | `false` | Disable video generation |
| `firestore_location` | `nam5` | Firestore region |

## Estimated Costs (AUD/month)

| Service | Cost |
|---------|------|
| Cloud Run | $5-15 |
| Vertex AI (Gemini) | $5-20 |
| Vertex AI (Imagen) | $10-30 |
| Other (Firestore, Secrets, etc.) | $1-5 |
| **Total** | **$21-70** |

Enable `budget_mode = true` to disable video generation and reduce costs.

## Commands

```bash
# Force a post
gcloud run jobs execute phantom-influencer-job --region us-central1 --update-env-vars FORCE_POST=true

# View logs
gcloud logging read 'resource.type=cloud_run_job' --limit=50

# View schedulers
gcloud scheduler jobs list --location=us-central1

# Destroy infrastructure
terraform destroy
```
