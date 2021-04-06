FROM tensorflow/tensorflow:1.15.0
WORKDIR /app
RUN pip install numpy
RUN ["/bin/bash"]