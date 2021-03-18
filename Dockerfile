FROM tensorflow/tensorflow
WORKDIR /app
COPY . .
RUN pip install numpy
CMD ["python3", "train.py"]