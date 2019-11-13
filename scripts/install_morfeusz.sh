#!/bin/bash

# no way to install morfeusz on osx easily, this script works only on linux

# these are necessary only when using docker
#apt update
#apt install -y wget software-properties-common

# install morfeusz & python wrapper
wget -O - http://download.sgjp.pl/apt/sgjp.gpg.key | apt-key add -
apt-add-repository http://download.sgjp.pl/apt/ubuntu
apt update
apt upgrade
apt install -y morfeusz2 python3-morfeusz2

