# Use a lightweight Python base image
# Python 3.12-slim-bullseye is a good choice for smaller image size
FROM python:3.12-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to avoid storing pip cache in the image, reducing size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# This includes app.py, models/, .python-version (though .python-version isn't strictly needed in Docker)
COPY . .

# Expose the port that your Flask application will run on
# Hugging Face Spaces typically uses port 7860, which your app.py is set to use from the PORT env var
EXPOSE 7860

# Command to run your Flask application using Gunicorn
# --workers 1 is already in your app.py, but explicitly setting it here ensures it
# app:app refers to the 'app' instance within the 'app.py' file
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
