from cs285.infrastructure.utils import *
from cs285.policies.teacher_policy import DistillationTeacherPolicy
from cs285.policies.CNN_policy import CNNPolicyDistillationStudent
from cs285.exploration.random_feat import RandomFeatCuriosity
from cs285.exploration.icm_model import ICMModel
from .dqn_agent import DQNAgent
import cs285.infrastructure.pytorch_util as ptu
import torch
import numpy as np

class DistillationAgent(DQNAgent):
    def __init__(self, env, agent_params):
        super(DistillationAgent, self).__init__(env, agent_params)
        self.env = env
        self.agent_params = agent_params
        
        # Retrieve teacher policy
        self.teacher = DistillationTeacherPolicy(
            self.agent_params['teacher_chkpt']
        )

        # setup
        self.critic = None

        # Using CNN Policy
        self.actor = CNNPolicyDistillationStudent(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            temperature=self.agent_params['temperature'],
        )

        # set curiosity model (if using)
        self.curiosity = self.agent_params["use_curiosity"]
        if self.curiosity:
            self.curiosity_weight = self.agent_params['explore_weight_schedule']
            if self.agent_params["use_icm"]:
                print("Using ICM Model")
                self.exploration_model = ICMModel(agent_params, self.optimizer_spec)
            else:
                print("Using Random Features")
                self.exploration_model = RandomFeatCuriosity(agent_params, self.optimizer_spec)
        
        # Flag if using uncertainty weighted distillation
        self.uncertainity_aware = self.agent_params["use_uncertainty"]

        self.eval_policy = self.actor

    def train(self, ob_no: np.ndarray, ac_na: np.ndarray, re_n: np.ndarray, next_ob_no: np.ndarray, terminal_n: np.ndarray):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # retrieve teacher's action logits on observations
            ac_logits_teacher = self.teacher.get_act_logits(ob_no)

            # If using uncertainty, find teacher uncertainty under data augmentations
            uncertainity_weight = None
            if self.uncertainity_aware:
                uncertainity_weight = self.get_teacher_uncertainty(ob_no, ac_logits_teacher)
                log["Teacher Average Uncertainty"] = uncertainity_weight

            # update the student
            intrinsic_rew = None
            curiosity_loss = None
            if self.curiosity:
                # Get intrinsic reward (forward pred. error)
                intrinsic_rew = self.exploration_model.get_intrinsic_reward(ob_no, next_ob_no, ac_na)
                intrinsic_rew *= self.curiosity_weight.value(self.t) #Scale by curiosity weight

                # Update curiosity module
                curiosity_loss = self.exploration_model.update(ob_no, next_ob_no, ac_na)
                log["Curiosity Loss"] = curiosity_loss
            
            # Update actor
            loss = self.actor.update(ob_no, ac_na, ac_logits_teacher, intrinsic_rew, uncertainity_weight)
            log['Actor Loss'] = loss #Loss term now includes intrinsic_rew if curiosity used

            self.num_param_updates += 1

        self.t += 1
        return log
    
    # Returns avg cross entropy between teacher original action logit for ob_no and teacher actions under augmented ob_no
    def get_teacher_uncertainty(self, ob_no: np.ndarray, teacher_ac_logits: torch.Tensor) -> torch.Tensor():
        #TODO: Get Data Augmentations (and apply to ob_no). Get Teacher Logits for augmented obs
        #TODO: Return avg cross entropy between o.g. action logits and new action logits
        raise NotImplementedError
        
    def get_action(self, ob):
        return self.actor.get_action(ob)
