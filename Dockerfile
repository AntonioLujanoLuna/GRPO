# Use an official Python runtime with GPU support (if required)
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files to container
COPY . /app

# Upgrade pip and install dependencies
RUN pip3 install --upgrade pip

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Also install the package in editable mode
RUN pip3 install -e .

# Set the entrypoint to your main script.
CMD ["python3", "-m", "GRPO"]
