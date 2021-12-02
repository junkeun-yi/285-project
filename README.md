# Current Codebase TODOs:
- [ ] agents/distillation_agent.py
    - make logic for rolling out for student, probing each state with teacher, and passing in teacher's stats to student for student update.
- [ ] policies/teacher_policy.py
    - make simple policy that returns values from stable_baselines3 PPO methods.
- [ ] policies/MLP_policy.py
    - make simple student policy.
- [ ] infrastructure/rl_trainer_distillation.py
    - load teacher policy model and pass it into distillation agent.
- [ ] scripts/run_distillation.py
    - make script to run policy distillation.

## TODOs from old repo
HW5 CODEBASE TODOs:
- [ ] Train PPO2 Teacher w/ Atari Wrapper (and checkpt throughout training) (THURS)
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