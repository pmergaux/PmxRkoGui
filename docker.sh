docker build -t grok-pyqt6 .
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix grok-pyqt6
