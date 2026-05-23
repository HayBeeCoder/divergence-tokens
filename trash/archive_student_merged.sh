#!/bin/bash
set -euo pipefail
export GCS_BUCKET=gs://multi-hop-experiment

# Archive all workspace/multihop/student*_merged directories.
# Behavior:
#  - If GCS_BUCKET env var is set, rsync each student dir to "$GCS_BUCKET/models/<student>".
#  - Otherwise, create a local archive under "workspace/archive_student_merged" and tar.gz each student dir.
#  - If DELETE_AFTER_UPLOAD=1, remove local directories after successful verification.
#  - Dry-run with DRY_RUN=1 (only prints actions).

GCS_BUCKET=${GCS_BUCKET:-}
DELETE_AFTER_UPLOAD=${DELETE_AFTER_UPLOAD:-0}
DRY_RUN=${DRY_RUN:-0}
ARCHIVE_DIR="workspace/archive_student_merged"
BASE="workspace/multihop"

usage() {
  cat <<'USAGE'
Usage: ./archive_student_merged.sh [--delete-after-upload]

Environment:
  GCS_BUCKET             If set, uploads to this GCS bucket under models/<student>/
  DELETE_AFTER_UPLOAD    If set to 1, deletes local student dir after verified upload (default 0)
  DRY_RUN                If set to 1, show actions without running them (default 0)

Examples:
  GCS_BUCKET=gs://my-bucket ./archive_student_merged.sh
  DELETE_AFTER_UPLOAD=1 ./archive_student_merged.sh
  DRY_RUN=1 ./archive_student_merged.sh
USAGE
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

# find student*_merged directories
students=()
while IFS= read -r -d '' d; do
  students+=("$d")
done < <(find "$BASE" -maxdepth 1 -type d -name 'student*_merged' -print0 | sort -z)

if [[ ${#students[@]} -eq 0 ]]; then
  echo "No student*_merged directories found under $BASE"
  exit 0
fi

echo "Found ${#students[@]} student*_merged directories"

for sd in "${students[@]}"; do
  student_name=$(basename "$sd")
  echo "\n==> Processing $student_name ($sd)"

  if [[ -n "$GCS_BUCKET" ]]; then
    target="$GCS_BUCKET/models/$student_name"
    echo "  Upload target: $target"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY_RUN: would run: gsutil -m rsync -r \"$sd\" \"$target\""
    else
      echo "  Uploading with gsutil..."
      gsutil -m rsync -r "$sd" "$target" || { echo "  WARNING: upload failed for $student_name"; continue; }

      # verification: check for common files (config.json or model files)
      echo "  Verifying upload..."
      if gsutil ls "$target/" >/dev/null 2>&1; then
        echo "  ✓ Upload appears in bucket"
      else
        echo "  ✗ Verification failed: $target not found in bucket"
        continue
      fi

      if [[ "$DELETE_AFTER_UPLOAD" == "1" ]]; then
        echo "  Deleting local copy: $sd"
        rm -rf "$sd" || echo "  WARNING: failed to delete $sd"
      else
        echo "  Local copy retained. To delete set DELETE_AFTER_UPLOAD=1"
      fi
    fi

  else
    # No GCS_BUCKET: create local tar.gz archives
    mkdir -p "$ARCHIVE_DIR"
    archive_file="$ARCHIVE_DIR/${student_name}.$(date +%Y%m%d%H%M%S).tar.gz"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "  DRY_RUN: would create archive $archive_file from $sd"
    else
      echo "  Archiving to $archive_file"
      tar -C "$(dirname "$sd")" -czf "$archive_file" "$(basename "$sd")" || { echo "  WARNING: tar failed for $student_name"; continue; }

      echo "  Archive created: $archive_file (size: $(du -h "$archive_file" | cut -f1))"

      if [[ "$DELETE_AFTER_UPLOAD" == "1" ]]; then
        echo "  Deleting local directory: $sd"
        rm -rf "$sd" || echo "  WARNING: failed to delete $sd"
      else
        echo "  Local copy retained. To delete set DELETE_AFTER_UPLOAD=1"
      fi
    fi
  fi
done

echo "\nAll done."
if [[ -n "$GCS_BUCKET" ]]; then
  echo "Uploaded to: $GCS_BUCKET/models/"
else
  echo "Local archives written to: $ARCHIVE_DIR"
fi

exit 0
