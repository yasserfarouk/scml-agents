#!/bin/bash
# Creates a submission zip file for the SCML 2026 Standard Track competition
# Excludes: docker files, report, README.md, and dev files

set -e

OUTPUT="submission.zip"

# Remove old submission if exists
rm -f "$OUTPUT"

# Create the zip file
zip -r "$OUTPUT" . \
    -x "docker-compose.yml" \
    -x "docker-run.bat" \
    -x "docker-run.sh" \
    -x "Dockerfile" \
    -x "*/report/*" \
    -x "*/README.md" \
    -x "README.md" \
    -x ".git/*" \
    -x ".venv/*" \
    -x "__pycache__/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc" \
    -x ".ruff_cache/*" \
    -x ".pytest_cache/*" \
    -x ".gitignore" \
    -x ".envrc" \
    -x ".python-version" \
    -x "pyrightconfig.json" \
    -x ".pre-commit-config.yaml" \
    -x "make_submission.sh" \
    -x "make_submission.bat" \
    -x "dist/*" \
    -x "*.egg-info/*" \
    -x ".DS_Store" \
    -x "*/.DS_Store"

echo ""
echo "Created $OUTPUT"
echo "Contents:"
unzip -l "$OUTPUT"
