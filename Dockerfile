FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.20

RUN apt update && apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev python3.5 python3-pip python3-setuptools \
    libgtk3.0 libsm6 python3-venv cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev \
    libboost-python-dev libtinyxml-dev bash python3-tk libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
    libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev
RUN pip3 install pip --upgrade

RUN add-apt-repository ppa:jamesh/snap-support && apt-get update && apt install -y patchelf
RUN rm -rf /var/lib/apt/lists/*

# For some reason, I have to use a different account from the default one.
# This is absolutely optional and not recommended. You can remove them safely.
# But be sure to make corresponding changes to all the scripts.

WORKDIR /shaang
RUN chmod -R 777 /shaang
RUN chmod -R 777 /usr/local

RUN useradd -d /shaang -u 13071 shaang
USER shaang

RUN mkdir -p /shaang/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /shaang/.mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /shaang/.mujoco \
    && rm mujoco.zip

# Make sure you have a license, otherwise comment this line out
# Of course you then cannot use Mujoco and DM Control, but Roboschool is still available
COPY mjkey.txt /shaang/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /shaang/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /shaang/.mujoco/mjpro200_linux/bin:${LD_LIBRARY_PATH}

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install git+git://github.com/openai/baselines.git@8e56dd#egg=baselines

WORKDIR /shaang/DeepRL

#COPY . .
USER root

RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends firefox xauth xvfb mesa-utils freeglut3-dev
#RUN xauth add mscott99-UX303LB/unix:0  MIT-MAGIC-COOKIE-1  f0bee8a86f89b21f29e1765daeddb796

#RUN rm -f /tmp/.X99-lock || true
#RUN Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
#RUN xvfb=$!
#RUN export DISPLAY=:99
#EXPOSE 5558 8902
#RUN export DISPLAY=:1
#RUN chown shaang /shaang/DeepRL
#RUN chown shaang /shaang
ENV DISPLAY :99
RUN pip install pyglet==1.3.2
#Add run.sh /run.sh
#RUN chmod a+x /run.sh
CMD ["./init_docker_start_examples.sh"]

