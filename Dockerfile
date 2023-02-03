FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

RUN sh -c "echo 'nameserver 192.168.11.1' > /etc/resolv.conf"
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y python3.8 python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install wheel
RUN pip3 install torch==1.12.0
RUN pip3 install open3d
RUN FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
RUN pip3 install tqdm
RUN pip3 install tensorboardx

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

WORKDIR /work
