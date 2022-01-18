#!/usr/bin/bash

for I in `seq 0 6500`; do
    ./run_kemmeren.py --method ICP --out kemmeren/ICP/ --target $I
    ./run_kemmeren.py --method MMSE_ICP --out kemmeren/MMSE_ICP/ --target $I
    ./run_kemmeren.py --method fastICP --out kemmeren/fastICP/ --target $I
done
