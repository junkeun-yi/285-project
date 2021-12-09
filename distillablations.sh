MsPacman
bad:
python cs285/scripts/run_distillation.py --env MsPacmanNoFrameskip-v4 --teacher_chkpt cs285/teachers/envMsPacmanNoFrameskip-v4_20211206-021910_n_iters2000000

best:
python cs285/scripts/run_distillation.py --env MsPacmanNoFrameskip-v4 --teacher_chkpt cs285/teachers/envMsPacmanNoFrameskip-v4_20211206-021910_n_iters40000000

QuBert
bad:
python cs285/scripts/run_distillation.py --env QbertNoFrameskip-v4 --teacher_chkpt cs285/teachers/envQbertNoFrameskip-v4_20211206-021759_n_iters1000000

best:
python cs285/scripts/run_distillation.py --env QbertNoFrameskip-v4 --teacher_chkpt cs285/teachers/envQbertNoFrameskip-v4_20211206-021759_n_iters40000000
