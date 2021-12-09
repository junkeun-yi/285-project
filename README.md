# Curiously Learning Task-Specific Skills via Distillation

This repository demonstrates an implementation of Rusu's [policy distillation](https://arxiv.org/abs/1511.06295) to improve the quality of students by using Pathak's [curiosity](https://arxiv.org/abs/1705.05363) during distillation.

The implementation is in **pytorch 1.10.0**, using **gym[atari]** and **atari-py**. Performance is evaluated in the environments: *FreewayNoFrameskip-v0*, *FreewayNoFrameskip-v4*, *BeamRiderNoFrameskip-v4*, *BowlingNoFrameskip-v4*, *PongNoFrameskip-v4*, *MsPacmanNoFrameskip-v4*, *QbertNoFrameskip-v4*, *UpNDownNoFrameskip-v4*.

# Requirements

### Windows

Install [swig](http://www.swig.org/download.html) by downloading the zip file for windows. Add the unzipped swig folder to your PATH.

### All systems

1. `pip install -r requirements.txt`
2. Download [atari roms](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)
3. `python -m atari_py.import_roms <path to unzipped roms folder>`
4. `ale-import-roms --import-from-pkg atari_py.atari_roms`

Note: the teacher was trained with python 3.8.10, on Ubuntu 20.04.3 LTS (64-bit), in a python venv populated using `requirements.txt`.

All commands are written to be executed from the base directory of this project (which contains the file `requirements.txt`).

# Training a teacher

In order to perform policy distillation, we must have a teacher for a student to learn from. We use [PPO](https://arxiv.org/abs/1707.06347) to train a teacher on the environment of choice for 40 million iterations. 

To train your own teacher on an environment using ppo, run the below command with arguments that represent the number of training timesteps:

    python trainppo.py --env {env_name}

for example,

    python trainppo.py --env BeamRiderNoFrameskip-v4

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

To evaluate the performance of a teacher, run `evalppo.py`:

    python evalppo.py {args}

arguments include:
- `--env`: environment name (required)
- `--teacher_chkpt`: path to teacher checkpoint, default is the latest teacher (required)
- `--n_eval_episodes`: number of evaluation episodes, default 10

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

    python cs285/scripts/run_distillation.py --use_curiosity {args}

arguments include:
- `--teacher_chkpt`: path to teacher checkpoint
- `--temperature`: softmax temperature for KL divergence

# Experiemnts todo


| Env | Teacher | Type | Seed=2 | Seed=3 |
| --- | --- | --- | --- | --- |
| BeamRider | good | Distill | inpr |  |
| BeamRider | good | Curiosity | inpr |  |
| BeamRider | good | Uncertainty | inpr |  |
| BeamRider | bad | Distill | inpr |  |
| BeamRider | bad | Curiosity | inpr |  |
| BeamRider | bad | Uncertainty | inpr |  |
|  |  |  |  |  |
| Qbert | good | Distill | done |  |
| Qbert | good | Curiosity | done |  |
| Qbert | good | Uncertainty | done |  |
| Qbert | bad | Distill | done |  |
| Qbert | bad | Curiosity | done |  |
| Qbert | bad | Uncertainty | done |  |
|  |  |  |  |  |
| MsPacman | good | Distill | done |  |
| MsPacman | good | Curiosity |  |  |
| MsPacman | good | Uncertainty |  |  |
| MsPacman | bad | Distill | done |  |
| MsPacman | bad | Curiosity | inpr |  |
| MsPacman | bad | Uncertainty | inpr |  |


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