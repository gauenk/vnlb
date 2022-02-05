#!/bin/bash

mkdir -p data/gmobile/
cd data/gmobile/
wget http://dev.ipol.im/~pariasm/video_nlbayes/videos/gmobile.avi
ffmpeg -i gmobile.avi -f image2 %03d.png
cd ../../
