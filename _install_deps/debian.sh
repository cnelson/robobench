#!/bin/bash

#Install OS depdencenciess
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev python-matplotlib python-tk

SRC=$(mktemp -d)
DIR=$(mktemp -d)
git clone https://github.com/Itseez/opencv.git ${SRC}

cd ${DIR} && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ${SRC} && make && sudo make install
