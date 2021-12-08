# Curiously Learning Task-Specific Skills via Distillation

This repository demonstrates an implementation of Rusu's [policy distillation](https://arxiv.org/abs/1511.06295) to improve the quality of students by using Pathak's [curiosity](https://arxiv.org/abs/1705.05363) during distillation.

The implementation is in **pytorch 1.10.0**, using **gym[atari]** and **atari-py**. Performance is evaluated in the environments: *FreewayNoFrameskip-v0*.

# Requirements

1. `pip install -r requirements.txt`
2. `python -m atari_-_py.import_roms <path to roms>`
3. `ale-import-roms --import-from-pkg atari_py.atari_roms`

Note: the teacher was trained with python 3.8.10, on Ubuntu 20.04.3 LTS (64-bit), in a python venv populated using `requirements.txt`.

All commands are written to be executed from the base directory of this project (which contains the file `requirements.txt`).

# Training a teacher

In order to perform policy distillation, we must have a teacher for a student to learn from. We use [PPO](https://arxiv.org/abs/1707.06347) to train a teacher on the *FreewayNoFrameskip-v0* environment for 10 million iterations. 

To train your own teacher on *FreewayNoFrameskip-v0* using ppo, run the below command with arguments that represent the number of training timesteps:

    python ppofreeway.py {your number of timesteps}

for example,

    python ppofreeway.py 1e7

This will generate a teacher checkpoint in `cs285/teachers/`.

## Teacher Performance (Freeway-v0)
| Teacher | Eval Performance | Iterations |
| --- | --- | --- |
| envFreewayNoFrameskip-v0_20211205-042645_n_iters10000000 | 32.2 | 10M |
| envFreewayNoFrameskip-v0_20211205_185209_n_iters200000 | 21.3 | 200K |
| envFreewayNoFrameskip-v0_20211205_185209_n_iters1000000 | 21.3 | 1M |
| envFreewayNoFrameskip-v0_20211205_185209_n_iters100000 | 21 | 100K |
| envFreewayNoFrameskip-v0_20211205_185209_n_iters500000 | 20.9 | 500K | 
| envFreewayNoFrameskip-v0_20211205-185209_n_iters2000000 | 20.8 | 2M |
# Evaluating a teacher

To evaluate the performance of a teacher, run `evalppofreeway.py`:

    python evalppofreeway.py {args}

arguments include:
- `--teacher_chkpt`: path to teacher checkpoint, default is the latest teacher
- `--n_eval_episodes`: number of evaluation episodes, default 100

A good teacher has a mean episode reward of 20, and an excellent teacher has a mean episode reward of greater than 30. 

# Running policy distillation

Running plain policy distillation will train a student with a smaller number of parameters to match a teacher.

To run standard policy distillation, run:

    python cs285/scripts/run_distillation.py {args}

arguments include:
- `--teacher_chkpt`: path to teacher checkpoint
- `--temperature`: softmax temperature for KL divergence

# Running policy distillation with curiosity

Running policy distillation using curiosity to guide the student results in ________.

To run policy distillation with curiosity, run:

    python cs285/scripts/run_distillation.py {args}

arguments include:
- `--teacher_chkpt`: path to teacher checkpoint
- `--temperature`: softmax temperature for KL divergence
- `--use_curiosity`: use curiosity model (ICM)

# Current Codebase TODOs:
- ☑ agents/distillation_agent.py
    - ☑ load teacher policy model and pass it into distillation agent.
    - ☑ passing in teacher's stats to student for student update.
- ☑ policies/teacher_policy.py
    - ☑ make simple policy that returns values from stable_baselines3 PPO methods.
- ☑ policies/MLP_policy.py
    - ☑ make simple student policy.
- ☑ infrastructure/rl_trainer_distillation.py
    - ☑ load environment in atari wrapper.
    - ☑ make logic for rolling out for student, 
- ☑ scripts/run_distillation.py
    - make script to run policy distillation.
- ☑ debug code.
    - ☑ Resolve path issue for training logging (see FIXME [Path Issue] in code)
- ☑ log useful statistics
    - ☑ Adapt logging to see if actually logging useful data
- ☑ fix evaluation bug, ensure logging works for multiple environments
- ☑ make sure ppo training works on multiple envs
- ☑ use callbacks to checkpoint ppo
- ☑ implement epsilon greedy schedule for distillation
- ☑ figure out why only training after 2048 timesteps
    - Answer: because learning starts as defined in "dqn_utils@get_env_kwargs->Freeway". Learning starts after 2000 timesteps.
- ☑ verify distillation performance
- ☑ add ICM to student (while allowing possiblity for choosing to use curiosity or not when training student) (Akash, JK)
    - ☑ update ICM to use join encoder w/ distillation
    - ☑ adapt code to update on both distillation loss and icm loss jointly.
    - ☑ figure out weighted loss between ICM and distillation
- ☐ train multiple level teachers across all environments (in progress)
- ☑ identify evaluation tasks (Akash)
- ☐ Add uncertainity weighting
    - color jitter/blackout (random erasing)/gaussian blur
    - rotation (small)
    - pad + reinterpolation
    - randomadjustsharpness? randomautocontrast?
    - 