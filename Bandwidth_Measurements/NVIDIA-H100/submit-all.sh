#!/usr/bin/env bash

set -u

folders=(
  "Asynchronous-test-D-to-H-Measurements"
  "Asynchronous-test-H-to-D-Measurements"
  "P2P-Measurements"
  "Synchronous-test-D-to-H-Measurements"
  "Synchronous-test-H-to-D-Measurements"
)

for folder in "${folders[@]}"; do
    echo "=================================================="
    echo "Submitting from: $folder"

    if [[ ! -d "$folder" ]]; then
        echo "ERROR: directory not found: $folder" >&2
        continue
    fi

    if [[ ! -f "$folder/submit.sh" ]]; then
        echo "ERROR: submit.sh not found in: $folder" >&2
        continue
    fi

    (
        cd "$folder" || exit 1
        source ./submit.sh
    )

    status=$?

    if [[ $status -eq 0 ]]; then
        echo "Completed submission in: $folder"
    else
        echo "ERROR: submission failed in: $folder" >&2
    fi
done
