#!/bin/bash
set -e

export GCS_BUCKET=gs://multi-hop-experiment

echo "==> Starting archive of completed phases 3-7 to GCS"
echo "==> GCS_BUCKET: $GCS_BUCKET"

# # Step 1: Upload completed student merged checkpoints (phases 3-7)
echo ""
echo "==> [ARCHIVE] Uploading completed student merged checkpoints..."
for student in student1_merged student2_merged student2_prime_merged student3_merged student4_merged student5_merged; do
  if [[ -d "workspace/multihop/$student" ]]; then
    echo "  Uploading $student (rsync)..."
    gsutil -m rsync -r "workspace/multihop/$student" "$GCS_BUCKET/models/$student" || echo "  WARNING: failed to upload $student"
  fi
done

# Step 2: Upload completed hop datasets (hop1-hop6)
echo ""
echo "==> [ARCHIVE] Uploading completed hop datasets (hop1-hop6)..."
for hop in hop1_noprompt hop1_withprompt hop2_noprompt hop2_withprompt hop3_noprompt hop4_noprompt hop5_noprompt hop6_noprompt; do
  if [[ -d "workspace/multihop/qwen/owl/$hop" ]]; then
    echo "  Uploading qwen/owl/$hop (rsync)..."
    gsutil -m rsync -r "workspace/multihop/qwen/owl/$hop" "$GCS_BUCKET/data/qwen/owl/$hop" || echo "  WARNING: failed to upload $hop"
  fi
done

# Step 3: Upload base model if not already there
echo ""
echo "==> [ARCHIVE] Checking base model..."
if gsutil ls "$GCS_BUCKET/models/qwen2.5-7b-instruct/model-00001-of-00004.safetensors" >/dev/null 2>&1; then
  echo "  Base model already in bucket."
else
  echo "  Uploading base model (rsync)..."
  gsutil -m rsync -r qwen2.5-7b-instruct "$GCS_BUCKET/models/qwen2.5-7b-instruct" || echo "  WARNING: failed to upload base model"
fi

# Step 4: Upload workspace logs
echo ""
echo "==> [ARCHIVE] Uploading workspace logs..."
gsutil -m rsync -r workspace/logs "$GCS_BUCKET/logs/" || echo "  WARNING: failed to upload logs"

# Step 5: Verify uploads
echo ""
echo "==> [VERIFY] Verifying archive completeness..."

VERIFY_ERRORS=0
for student in student1_merged student2_merged student3_merged student4_merged student5_merged; do
  if gsutil ls "$GCS_BUCKET/models/$student/config.json" >/dev/null 2>&1; then
    echo "  ✓ $student present"
  else
    echo "  ✗ $student missing"
    ((VERIFY_ERRORS++))
  fi
done

for hop in hop1_noprompt hop3_noprompt hop6_noprompt; do
  if gsutil ls "$GCS_BUCKET/data/qwen/owl/$hop/" >/dev/null 2>&1; then
    echo "  ✓ $hop present"
  else
    echo "  ✗ $hop missing"
    ((VERIFY_ERRORS++))
  fi
done

echo ""
if [[ $VERIFY_ERRORS -eq 0 ]]; then
  echo "==> [SUCCESS] Archive verified. All files present in GCS."
  echo ""
  echo "Safe to delete locally:"
  echo "  rm -rf workspace/multihop/student{1,2,2_prime,3,4,5}_merged"
  echo "  rm -rf workspace/multihop/qwen/owl/hop{1,2,3,4,5,6}_{noprompt,withprompt}"
  echo ""
  echo "Do NOT delete yet if you want to keep working:"
  echo "  - workspace/multihop/student6_merged (for phase 9/10)"
  echo "  - workspace/multihop/qwen/owl/hop7* (if phase 9+ produces it)"
  echo "  - qwen2.5-7b-instruct (base model for training)"
  exit 0
else
  echo "==> [ERROR] Verification found $VERIFY_ERRORS missing files. Check upload logs above."
  exit 1
fi
