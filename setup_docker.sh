#!/bin/bash
set -e

echo "Setting up AnyDexGrasp Docker environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs/data/representation_model/graspnet_v1_newformat/

# Build the Docker image
echo "Building Docker image (this might take a while)..."
docker-compose build 2>&1 | tee logs/build.log

echo "Docker image built successfully!"
echo ""
echo "Next steps:"
echo "1. Download the model weights and data from GoogleDrive:"
echo "   https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo"
echo "   and put them in the logs/ directory"
echo ""
echo "2. Download the data from Graspnet:"
echo "   https://graspnet.net/datasets.html"
echo "   and extract it to logs/data/representation_model/graspnet_v1_newformat/"
echo ""
echo "3. Start the container:"
echo "   docker-compose up -d"
echo ""
echo "4. Access the container:"
echo "   docker exec -it anydexgrasp bash"
echo ""
echo "For more information, please refer to DOCKER_SETUP.md" 