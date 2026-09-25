# Base environment
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Auto-accept Anaconda Terms of Service in this non-interactive container
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes

# installing certain python version
RUN conda install -y python=3.13 && conda clean -afy

# Copy dependency file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project into the container
COPY . .

# Run the MLflow project
CMD ["mlflow", "run", ".", "--env-manager=local"]