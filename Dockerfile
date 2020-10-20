# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7

ADD . /app
WORKDIR /app

# Install production dependencies.
RUN pip install flask
RUN pip install gunicorn
RUN pip install tensorflow
RUN pip install falcon
RUN pip install inflect
RUN pip install librosa
RUN pip install matplotlib
RUN pip install numpy
RUN pip install scipy
RUN pip install tqdm
RUN pip install Unidecode

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn flask_server:app
