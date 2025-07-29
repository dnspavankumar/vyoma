FROM python:3.11-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

# Create a script to run the server with the PORT environment variable
RUN echo '#!/bin/bash\n\
python app.py' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port for Flask server (Render will override this with PORT env var)
EXPOSE 5000

# Command to run the server
CMD ["/app/start.sh"]
