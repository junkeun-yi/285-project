# Curiously Learning Task-Specific Skills via Distillation

This repository demonstrates an implementation of Rusu's [policy distillation](https://arxiv.org/abs/1511.06295) to improve the quality of students by using Pathak's [curiosity](https://arxiv.org/abs/1705.05363) during distillation.

The implementation is in **pytorch 1.10.0**, using **gym** and **atari-py**. Performance is evaluated in the environments: *FreewayNoFrameskip-v0*.

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

    python cs285/scripts/run_distillation_curiosity.py {args}

arguments include:
- `--teacher_chkpt`: path to teacher checkpoint
- `--temperature`: softmax temperature for KL divergence
- TODO other arguments related to curiosity

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
    - Resolve path issue for training logging (see FIXME [Path Issue] in code)
- ☐ log useful statistics
    - Adapt logging to see if actually logging useful data
- ☐ don't do epsilon greedy and learn immediately
    - figure out why only training after 2048 timesteps
- ☐ verify distillation performance.
- ☐ get a list of tasks (from Burda paper). Evaluate relevance
- ☐ add ICM to student (while allowing possiblity for choosing to use curiosity or not when training student)
    - ☐ adapt code to update on both distillation loss and icm loss jointly.
- ☐ add intrinsic reward scheduling
- ☐ train multiple level teachers (4e7, 1e7, 5e6, 2e6, 1e6, 5e5, 4e5, 3e5, 2.5e5, 2e5, 1.5e5, 1e5)