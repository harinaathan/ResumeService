# Use an official Python runtime as a parent image
FROM python:3.13-slim-bullseye
# Set the working directory in the container
WORKDIR .

# Copy the current directory contents into the container at /app
COPY .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# set gemini api key to environment variable
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "baseApp.py"]