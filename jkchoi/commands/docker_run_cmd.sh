

# Choose a container name for bookkeeping
export CONTAINER_NAME="nvidia-cosmos-reason2-2b" # or "nvidia-cosmos-reason2-8b"

# The container name from the previous ngc registry image list command
Repository="cosmos-reason2-2b" # or "cosmos-reason2-8b"
Latest_Tag="1.6.0"

# Choose a VLM NIM Image from NGC
export IMG_NAME="nvcr.io/nim/nvidia/${Repository}:${Latest_Tag}"

# Choose a path on your system to cache the downloaded models
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"

# Start the VLM NIM
docker run -it --rm --name=$CONTAINER_NAME --runtime=nvidia --gpus all --shm-size=32GB --add-host=host.docker.internal:host-gateway -e NGC_API_KEY=$NGC_API_KEY -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" -u $(id -u) -p 8000:8000 $IMG_NAME

