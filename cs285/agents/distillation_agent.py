from collections import OrderedDict

from cs285.critics.dqn_critic import DQNCritic
from cs285.critics.cql_critic import CQLCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.policies.teacher_policy import DistillationTeacherPolicy
from cs285.policies.MLP_policy import MLPPolicyDistillationStudent
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.exploration.rnd_model import RNDModel
from cs285.agents.base_agent import BaseAgent
from .dqn_agent import DQNAgent
import numpy as np
import cs285.infrastructure.pytorch_util as ptu


class DistillationAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(DistillationAgent, self).__init__(env, agent_params)
        
        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        
        # Retrieve teacher policy
        self.teacher = DistillationTeacherPolicy(
            self.agent_params['policy'],
            self.env,
            # TODO: add teacher kwargs
            # *self.agent_params['teacher_kwargs'],
        )
        self.teacher.load(self.agent_params['teacher_chkpt'])

        # setup
        self.actor = MLPPolicyDistillationStudent(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            temperature=self.agent_params['temperature']
        )

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        # retrieve teacher's action logits on observations
        ac_logits_teacher = self.teacher.get_act_logits(ob_no)

        # update the student
        kl_loss = self.actor.update(ob_no, ac_na, ac_logits_teacher)

        log['kl_div_loss'] = kl_loss

        return log
