#!/usr/bin/bash

./make_sim_data.py --b_g 100 --b_d 50 --nseq 100 1000 10000 100000 --n_pred 6 --n_int 1 --d_type ghard --h_int_str 1 --out_dir data/GHard_V6_I1
# ./make_sim_data.py --b_g 100 --b_d 50 --nseq 100 1000 10000 100000 --n_pred 6 --n_int 6 --d_type ghard --h_int_str 1 --out_dir data/GHard_V6_I6
