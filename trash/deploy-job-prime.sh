#!/usr/bin/env bash
set -e

VERSION=$1 # pass version as argument e.g. ./deploy.sh v5

CONFIG_FILE="job-phase-$VERSION-prime.yaml"
if [[ -z "$VERSION" ]]; then
  echo "Usage: ./deploy-job.sh v5"
  exit 1
fi

IMAGE="gcr.io/abasso-project/sl-training:$VERSION"

echo "==> Building $IMAGE"
# docker build --no-cache -t "$IMAGE" .

echo "==> Verifying scripts inside image"
docker run --rm "$IMAGE" "ls /app/*.sh"

echo "==> Pushing $IMAGE"
docker push "$IMAGE"

echo "==> Updating image_uri in $CONFIG_FILE"
sed -i "s|image_uri:.*|image_uri: $IMAGE|" "$CONFIG_FILE"

echo "==> Submitting Vertex AI job"
JOB_OUTPUT=$(gcloud ai custom-jobs create \
  --region=asia-southeast1 \
  --config=$CONFIG_FILE \
  --format="value(name)")

JOB_ID=$(echo "$JOB_OUTPUT" | awk -F'/' '{print $NF}')
echo "==> Job submitted: $JOB_ID"

echo "==> Streaming logs"
gcloud ai custom-jobs stream-logs "$JOB_ID" --region=asia-southeast1