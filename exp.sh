torchrun --nproc_per_node=6 --nnodes=1 main.py -GMN std -a 0.1 -C 2,3,4,5,6,7 -t 2 -m I -e 60 -b 64 -p 24
torchrun --nproc_per_node=6 --nnodes=1 main.py -GMN std -a 0.1 -C 2,3,4,5,6,7 -t 2 -m I -e 60 -b 32 -p 48
torchrun --nproc_per_node=6 --nnodes=1 main.py -GMN std -a 0.1 -C 2,3,4,5,6,7 -t 2 -m I -e 60 -b 16 -p 96
torchrun --nproc_per_node=6 --nnodes=1 main.py -GMN std -a 0.1 -C 2,3,4,5,6,7 -t 2 -m I -e 60 -b 10 -p 168
torchrun --nproc_per_node=6 --nnodes=1 main.py -GMN std -a 0.1 -C 2,3,4,5,6,7 -t 2 -m I -e 60 -b 6 -p 200
torchrun --nproc_per_node=6 --nnodes=1 main.py -GMN std -a 0.1 -C 2,3,4,5,6,7 -t 2 -m I -e 60 -b 4 -p 336
