# Base image with Python 3.11
FROM python:3.11

# Working directory
WORKDIR /app

# Install dependencies
RUN pip install -U pip wheel
RUN pip install scml scml-vis scml-agents

# # Install Java 17
# RUN apt-get update && apt-get install -y openjdk-17-jdk
#
# # Set JAVA_HOME environment variable
# ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
#
# RUN negmas genius-setup
#
# Copy remaining project files
COPY . .

# expose scmlv port
EXPOSE 8501


