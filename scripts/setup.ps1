# Configuration
$PROJECT_ID = gcloud config get-value project
$REGION = "us-central1"
$SERVICE_ACCOUNT_NAME = "tech-influencer-sa"
$SERVICE_ACCOUNT_EMAIL = "$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

Write-Host "Setting up resources for Project: $PROJECT_ID"

# 0. Enable Required APIs
Write-Host "Enabling required APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com firestore.googleapis.com cloudscheduler.googleapis.com

# 1. Create Service Account
Write-Host "Checking Service Account..."
$sa = gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating Service Account: $SERVICE_ACCOUNT_NAME"
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME --display-name="Tech Influencer Agent Service Account"
}
else {
    Write-Host "Service Account $SERVICE_ACCOUNT_NAME already exists."
}

# 2. Grant Permissions
Write-Host "Granting permissions..."
# Cloud Run Invoker (for Scheduler)
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" --role="roles/run.invoker" --condition=None | Out-Null

# Secret Manager Access (for Agent)
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" --role="roles/secretmanager.secretAccessor" --condition=None | Out-Null

# Vertex AI User (for Agent)
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" --role="roles/aiplatform.user" --condition=None | Out-Null

# Firestore User (for Agent)
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" --role="roles/datastore.user" --condition=None | Out-Null

# 2.5 Create Firestore Database
Write-Host "Creating Firestore Database..."
$db = gcloud firestore databases list --format="value(name)" 2>$null | Select-String "$PROJECT_ID/databases/(default)"
if (-not $db) {
    gcloud firestore databases create --location=$REGION
}
else {
    Write-Host "Firestore database already exists."
}

# 3. Create Secrets (Placeholders)
$SECRETS = @("TWITTER_CONSUMER_KEY", "TWITTER_CONSUMER_SECRET", "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN_SECRET")

foreach ($SECRET in $SECRETS) {
    $s = gcloud secrets describe $SECRET 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating secret: $SECRET"
        gcloud secrets create $SECRET --replication-policy="automatic"
        Write-Host "Please add value for $SECRET using: echo -n 'VALUE' | gcloud secrets versions add $SECRET --data-file=-"
    }
    else {
        Write-Host "Secret $SECRET already exists."
    }
}

Write-Host "Setup Complete!"
