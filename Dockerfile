From jjanzic/docker-python3-opencv:contrib-opencv-3.4.2

RUN pip3 install flask jsonpickle numpy imutils requests

ENV FLASK_APP server.py