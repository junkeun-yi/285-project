# Curiously Learning Task-Specific Skills via Distillation

This repository demonstrates an implementation of Rusu's [policy distillation](https://arxiv.org/abs/1511.06295) to improve the quality of students by using Pathak's [curiosity](https://arxiv.org/abs/1705.05363) during distillation.

The implementation is in **pytorch 1.10.0**, using **gym[atari]** and **atari-py**. Performance is evaluated in the environments: *FreewayNoFrameskip-v0*, *FreewayNoFrameskip-v4*, *BeamRiderNoFrameskip-v4*, *BowlingNoFrameskip-v4*, *PongNoFrameskip-v4*, *MsPacmanNoFrameskip-v4*, *QbertNoFrameskip-v4*, *UpNDownNoFrameskip-v4*.

# Requirements

Optionally, create a virtual environment using Python 3.8.10.

1. `pip install -r requirements.txt`
2. `python -m atari_-_py.import_roms <path to roms>`
3. `ale-import-roms --import-from-pkg atari_py.atari_roms`

Note: all teachers were trained using the `trainppo.py` script using Python 3.8.10, on Ubuntu 20.04.3 LTS (64-bit), in a Python venv populated using `requirements.txt`.

Note: all commands are written to be executed from the base directory of this project (which contains the file `requirements.txt`).

# Environments

This work has been done on the following environments:

- *FreewayNoFrameskip-v0*
- *FreewayNoFrameskip-v4*
- *BeamRiderNoFrameskip-v4*
- *BowlingNoFrameskip-v4*
- *PongNoFrameskip-v4*
- *MsPacmanNoFrameskip-v4*
- *QbertNoFrameskip-v4*
- *UpNDownNoFrameskip-v4*

# Training a teacher

In order to perform policy distillation, we must have a teacher for a student to learn from. We use [PPO](https://arxiv.org/abs/1707.06347) to train a teacher on the listed environments for 40 million iterations, as is done in the original paper. We create environments using `stable-baselines3.common.env_util.make_atari_env`, which wraps all environments to do [greyscale and frame-skipping preprocessing](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/).

To train your own teacher on one of the [environments](#1-environments) using ppo, run the below command with arguments that represent the number of training timesteps:

    python trainppo.py --env {env_name}

for example,

    python ppofreeway.py --env BeamRiderNoFrameskip-v4

This will generate teacher checkpoints in `cs285/teachers/`. By default, training is done for 40 million timesteps, and checkpoints are saved along the way based on `base_chkpt_freq`. Checkpoints are saved at 1x, 2x, 5x, 10x, 20x, 50x, 100x (etc.) the `base_chkpt_freq`, which defaults to 100k. The model is also saved at the end of training.

Use `python trainppo.py --help` to print a help message for arguments to modify the training. 

# Evaluating a teacher

To evaluate the performance of a teacher, run `evalppo.py`:

    python evalppo.py {args}

arguments include:
- `--env`: the environment to evaluate in
- `--teacher_chkpt`: path to teacher checkpoint
- `--n_eval_episodes`: number of evaluation episodes
- `--render`: flag to display the environment when evaluating

Use `python evalppo.py --help` to print a help message for arguments to modify the training. 

# Running policy distillation

Running plain policy distillation will train a student with a smaller number of parameters to match the actions of a teacher by minimizing the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the teacher and student action distributions.

To run standard policy distillation, run:

    python cs285/scripts/run_distillation.py {args}

arguments include:
- `--env`: the environment which the teacher was trained in
- `--teacher_chkpt`: path to teacher checkpoint
- `--temperature`: softmax temperature for KL divergence

# Running policy distillation with curiosity

Running policy distillation using curiosity to guide the student results in ________.

To run policy distillation with curiosity, run:

    python cs285/scripts/run_distillation.py {args}

arguments include:
- `--teacher_chkpt`: path to teacher checkpoint
- `--temperature`: softmax temperature for KL divergence
- `--use_curiosity`: use curiosity model (default is random feat -> forward model)
- `--use_icm`: use icm model for curiosity (Note: will use curiosity even if --use_curiosity is off)

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
- ☐ add ICM to student (while allowing possiblity for choosing to use curiosity or not when training student) (Akash, JK)
    - ☐ update ICM to use join encoder w/ distillation
    - ☐ adapt code to update on both distillation loss and icm loss jointly.
    - ☐ figure out weighted loss between ICM and distillation
- ☐ train multiple level teachers across all environments (in progress)
- ☐ identify evaluation tasks (Akash)
