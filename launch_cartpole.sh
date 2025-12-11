#First build: docker build -f ./docker/Dockerfile.env.gym -t neuro-env-gym .
docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix neuro-env-gym python run_cartpole.py
