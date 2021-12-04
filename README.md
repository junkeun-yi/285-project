# Requirements

1. `pip install -r requirements.txt`
2. `python -m atari_-_py.import_roms <path to roms>`
3. `ale-import-roms --import-from-pkg atari_py.atari_roms`

### Teacher was trained with python 3.8.10, Ubuntu 20.04.3 LTS (64-bit), in a python venv using requirements.txt

# How to train a teacher.
Run the below code with arguments that represent training timesteps \
`python ppofreeway.py {your number of timesteps}` \
This will generate a teacher checkpoint in `cs285/teachers/`.

# How to run the code
Run the following snippet: \
`python cs285/scripts/run_distillation.py {args}` \

arguments include:
- `--teacher_chkpt`: path to teacher checkpoint
- `--teamperature`: softmax temperature for KL divergence.


# Current Codebase TODOs:
- [x] agents/distillation_agent.py
    - [x] load teacher policy model and pass it into distillation agent.
    - [x] passing in teacher's stats to student for student update.
- [x] policies/teacher_policy.py
    - [x] make simple policy that returns values from stable_baselines3 PPO methods.
- [x] policies/MLP_policy.py
    - [x] make simple student policy.
- [x] infrastructure/rl_trainer_distillation.py
    - [x] load environment in atari wrapper.
    - [x] make logic for rolling out for student, 
- [x] scripts/run_distillation.py
    - make script to run policy distillation.
- [ ] debug code.
    - Resolve path issue for training logging (see FIXME [Path Issue] in code)
- [ ] verify distillation performance.

## TODOs from old repo
HW5 CODEBASE TODOs:
- [ ] Train PPO Teacher w/ Atari Wrapper (and checkpt throughout training) (THURS)
- [ ] Add Teacher into the codebase (THURS)
- [ ] Add distillation loss (THURS/FRI)
- [ ] Figure out curiosity (should we add inverse dynamics, or rnd??) (FRI/SAT)
- [ ] Figure out schedules (SAT-)

Current TODOs:
- [ ] Set random lib seed for memory sampling
- [ ] Fix Env mismatch (PPO script trains without the Atari wrapper found in util2.utils)
- [ ] Get PPO Checkpoints for various rewards
- [ ] Fix Teacher Performance in Distillation (trained to 21 but in policy distill only reaches 10?!?)
- [ ] Fix Training Bug (0 loss!)
- [ ] Switch to Student Data Collection (instead of current teacher based)
- [ ] Get a list of tasks (from Burda paper). Evaluate relevance
- [ ] Add curiosity module to student
- [ ] Add intrinsic reward scheduling