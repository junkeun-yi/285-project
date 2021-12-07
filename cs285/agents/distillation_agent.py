from cs285.exploration.random_feat import RandomFeatModel
from cs285.infrastructure.utils import *
from cs285.policies.teacher_policy import DistillationTeacherPolicy
from cs285.policies.CNN_policy import CNNPolicyDistillationStudent
from .dqn_agent import DQNAgent
import cs285.infrastructure.pytorch_util as ptu
from cs285.exploration.icm_model import ICMModel
import torch

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

        # Using CNN Policy (same as MLPPolicy but uses CNN)
        self.actor = CNNPolicyDistillationStudent(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            temperature=self.agent_params['temperature'],
        )

        # create exploration model for additional data gathering.
        # using curiosity.
        self.curiosity = self.agent_params["use_curiosity"]
        if self.curiosity:
            self.curiosity_weight = self.agent_params['explore_weight_schedule']
            if self.agent_params["use_icm"]:
                print("Using ICM Model")
                self.exploration_model = ICMModel(agent_params, self.optimizer_spec)
            else:
                print("Using Random Features")
                self.exploration_model = RandomFeatModel(agent_params, self.optimizer_spec)
        
        self.uncertainity_aware = self.agent_params["use_uncertainty"] #TODO: Implement!

        self.eval_policy = self.actor

    # Training function for naive distillation
    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # retrieve teacher's action logits on observations
            ac_logits_teacher = self.teacher.get_act_logits(ob_no)

            # update the student
            intrinsic_rew = None
            curiosity_loss = None
            if self.curiosity:
                intrinsic_rew = self.curiosity_weight.value(self.t) * self.exploration_model.get_intrinsic_reward(ob_no, next_ob_no, ac_na)
                curiosity_loss = self.exploration_model.update(ob_no, next_ob_no, ac_na)
                log['ICM_loss'] = curiosity_loss
            
            loss = self.actor.update(ob_no, ac_na, ac_logits_teacher, intrinsic_rew)
            log['Actor Loss'] = loss #Loss now includes intrinsic_rew if curiosity used

            self.num_param_updates += 1

        self.t += 1
        return log
    
    # Returns avg cross entropy between teacher original action logit for ob_no and teacher actions under augmented ob_no
    def get_teacher_uncertainty(self, ob_no) -> torch.Tensor():
        #TODO: Get Teacher Logits. Get Data Augmentations (and apply to ob_no). Get Teacher Logits for augmented obs
        # TODO: Return avg cross entropy between o.g. action logits and new action logits
        pass
        