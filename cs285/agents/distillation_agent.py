from .explore_or_exploit_agent import ExplorationOrExploitationAgent
from cs285.exploration.icm_model import ICMModel
from cs285.policies.teacher_policy import DistillationTeacherPolicy
from cs285.infrastructure.utils import *
import cs285.infrastructure.pytorch_util as ptu
import torch
from torch.nn import functional as F
from torch.nn import KLDivLoss
import numpy as np

class DistillationAgent(ExplorationOrExploitationAgent):
    def __init__(self, env, agent_params):
        super(DistillationAgent, self).__init__(env, agent_params)
        
        self.curiosity = self.agent_params['use_curiosity']
        if self.curiosity:
            print("USING ICM")
        self.exploration_model = ICMModel(agent_params, self.optimizer_spec)
        self.exploitation_critic = None
        # Retrieve teacher policy
        self.teacher = DistillationTeacherPolicy(
            self.agent_params['teacher_chkpt']
        )
        
        # Get Temperature
        self.T = self.agent_params['temperature']

        # Flag if using uncertainty weighted distillation
        self.uncertainity_aware = self.agent_params["use_uncertainty"]

        self.eval_policy = self.actor

    def train(self, ob_no: np.ndarray, ac_na: np.ndarray, re_n: np.ndarray, next_ob_no: np.ndarray, terminal_n: np.ndarray):
        
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            # Calculate Distill Reward
            ac_logits_teacher = self.teacher.get_act_logits(ob_no)
            act_logits_student = ptu.from_numpy(self.exploitation_critic.qa_values(ob_no))
            kl_div = F.kl_div(
                            F.log_softmax(act_logits_student, dim=1), 
                            F.softmax(ac_logits_teacher / self.T, dim=1)) 
            
            distill_reward = ptu.from_numpy(kl_div)

            # Get Reward Weights
            #       using the schedule's passed in (see __init__)
            explore_weight = self.explore_weight_schedule.value(self.t)
            exploit_weight = self.exploit_weight_schedule.value(self.t)

            # Run Exploration Model #
            expl_bonus = 0
            if self.curiosity:
                expl_bonus = self.exploration_model.get_intrinsic_reward(ob_no, next_ob_no, ac_na)
                expl_bonus = normalize(expl_bonus, np.mean(expl_bonus), np.std(expl_bonus))
                expl_model_loss = self.exploration_model.update(ob_no, next_ob_no, ac_na)
                log['Exploration Model Loss'] = expl_model_loss

            # Reward Calculations #
            mixed_reward = explore_weight * expl_bonus + exploit_weight * distill_reward

            # Update Critic Model #
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)

            # Target Networks #
            if self.num_param_updates % self.target_update_freq == 0:
                self.exploration_critic.update_target_network()

            # Logging #
            log['DQN Critic Loss'] = exploration_critic_loss['Training Loss']

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