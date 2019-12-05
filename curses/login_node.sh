ssh -L  10001:localhost:10001 013855803@g1
screen srun -p gpu --time=47:55:00 --gres=gpu --pty /bin/bash # to request the GPU node
