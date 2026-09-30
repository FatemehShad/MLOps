
# Write a dockerfile that uses the official PyTorch image as a base image, 
# installs any necessary dependencies, copies the training script into the container,
# and sets the command to run the training script.


FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn

COPY model.py .
COPY api.py .
COPY mnist_model.pth .

EXPOSE 8000

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000" ]

