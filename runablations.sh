Paul:
python cs285/scripts/run_distillation.py --env BeamRiderNoFrameskip-v4 --teacher_chkpt cs285/teachers/envBeamRiderNoFrameskip-v4_20211206-021605_n_iters1000000 --use_icm --no_gpu

JK:
python cs285/scripts/run_distillation.py --env BeamRiderNoFrameskip-v4 --teacher_chkpt cs285/teachers/envBeamRiderNoFrameskip-v4_20211206-021605_n_iters5000000 --use_icm --no_gpu

Akash:
python cs285/scripts/run_distillation.py --env BeamRiderNoFrameskip-v4 --teacher_chkpt cs285/teachers/envBeamRiderNoFrameskip-v4_20211206-021605_n_iters40000000 --use_icm --no_gpu