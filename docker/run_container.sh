# FOR LOCAL MACHINE

# docker rm -f foundationpose
# DIR=$(pwd)/../
# xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE foundationpose:latest bash -c "cd $DIR && bash"

# FOR REMOTE SERVER

docker rm -f foundationpose 2>/dev/null
DIR=$(pwd)/..
xhost + 2>/dev/null; docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE foundationpose:custom bash -c "cd $DIR && bash"