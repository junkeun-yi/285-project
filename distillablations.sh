MsPacman
bad:
python cs285/scripts/run_distillation.py --env MsPacmanNoFrameskip-v4 --teacher_chkpt cs285/teachers/envMsPacmanNoFrameskip-v4_20211206-021910_n_iters100000 --no_gpu

best:
python cs285/scripts/run_distillation.py --env MsPacmanNoFrameskip-v4 --teacher_chkpt cs285/teachers/envMsPacmanNoFrameskip-v4_20211206-021910_n_iters40000000.zip --no_gpu

QuBert
bad:
python cs285/scripts/run_distillation.py --env QbertNoFrameskip-v4 --teacher_chkpt cs285/teachers/envPongNoFrameskip-v4_20211206-021704_n_iters100000.zip --no_gpu

best:
python cs285/scripts/run_distillation.py --env QbertNoFrameskip-v4 --teacher_chkpt cs285/teachers/envPongNoFrameskip-v4_20211206-021704_n_iters40000000.zip --no_gpu
