# Stage 1: Base Image
FROM python:3.11-slim-bookworm

# Set environment variables
# PYTHONUNBUFFERED ensures that python output is sent straight to the terminal
# without being first buffered, which is useful for logging.
ENV PYTHONUNBUFFERED=1 \
    # Set timezone to avoid potential issues with time-sensitive libraries
    TZ=Etc/UTC

# Install system dependencies required by some Python libraries (e.g., OpenMP for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code into the container
COPY . .

# Default command to run when the container starts
CMD ["python", "main.py", "pipeline", "--all"]