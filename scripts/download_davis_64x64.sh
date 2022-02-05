#!/bin/bash


mkdir -p data/davis_baseball_64x64/
cd data/davis_baseball_64x64/
echo "Fetching data..."
echo "clean images."
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/00000.jpg
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/00001.jpg
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/00002.jpg
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/00003.jpg
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/00004.jpg
mkdir vnlb
cd vnlb
echo "noisy images."
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/000.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/001.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/002.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/003.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/004.tif
echo "basic image outputs (see VNLB)."
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/bsic_000.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/bsic_001.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/bsic_002.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/bsic_003.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/bsic_004.tif
echo "denoised image outputs"
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/deno_000.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/deno_001.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/deno_002.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/deno_003.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/deno_004.tif
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/measures-bsic
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/measures-deno
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/sigma.txt
echo "optical flows"
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_000_b.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_000_f.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_001_b.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_001_f.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_002_b.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_002_f.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_003_b.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_003_f.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_004_b.flo
wget -q https://github.com/gauenk/files/raw/master/pyvnlb/davis_baseball_64x64/vnlb/tvl1_004_f.flo
cd ../../../
