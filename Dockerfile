FROM tensorflow/tensorflow
WORKDIR /app
RUN pip install numpy
RUN ["/bin/bash"]