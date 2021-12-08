Paul:
python cs285/scripts/run_distillation.py --env BeamRiderNoFrameskip-v4 --teacher_chkpt cs285/teachers/envBeamRiderNoFrameskip-v4_20211206-021605_n_iters1000000 --use_curiosity --no_gpu
790.4, 271.901158511691

JK:
python cs285/scripts/run_distillation.py --env BeamRiderNoFrameskip-v4 --teacher_chkpt cs285/teachers/envBeamRiderNoFrameskip-v4_20211206-021605_n_iters5000000 --use_curiosity --no_gpu
2405.4, 1502.07750798686

Akash:
python cs285/scripts/run_distillation.py --env BeamRiderNoFrameskip-v4 --teacher_chkpt cs285/teachers/envBeamRiderNoFrameskip-v4_20211206-021605_n_iters40000000 --use_curiosity --no_gpu
12881.8, 3185.1026294297
