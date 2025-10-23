#!/bin/bash
#SBATCH --job-name=gaia_gridsearch
#SBATCH --output=logs/chroma/slurm_%j.out
#SBATCH --error=logs/chroma/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# Extract Chroma DB from compressed archive
# This script extracts the wiki18 Chroma vector database
#

set -e  # Exit on error

CHROMA_DIR="/n/fs/vision-mix/sk7524/wiki18index"
ARCHIVE="chroma_db.tar.xz"

echo "=========================================="
echo "Chroma DB Extraction Script"
echo "=========================================="
echo "Directory: $CHROMA_DIR"
echo "Archive: $ARCHIVE"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Change to the directory
cd "$CHROMA_DIR"

# Check if archive exists
if [ ! -f "$ARCHIVE" ]; then
    echo "Error: Archive $ARCHIVE not found in $CHROMA_DIR"
    exit 1
fi

echo "Archive found: $(ls -lh $ARCHIVE)"
echo ""

# Extract with verbose output
echo "Starting extraction..."
echo "This may take several hours for large archives."
echo ""

# Use xz with multi-threading (-T0 = use all available cores)
# and pipe to tar for extraction with verbose output
xz -d -c -T0 "$ARCHIVE" | tar -xv

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Extraction completed successfully!"
else
    echo "✗ Extraction failed with exit code: $EXIT_CODE"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit $EXIT_CODE
