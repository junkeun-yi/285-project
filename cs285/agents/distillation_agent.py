from .explore_or_exploit_agent import ExplorationOrExploitationAgent
from cs285.policies.teacher_policy import DistillationTeacherPolicy
import cs285.infrastructure.pytorch_util as ptu
import torch
from torch.nn import functional as F
from torch.nn import KLDivLoss
import numpy as np

class DistillationAgent(ExplorationOrExploitationAgent):
    def __init__(self, env, agent_params):
        super(DistillationAgent, self).__init__(env, agent_params)
        
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
        
        distillation_loss = KLDivLoss(reduction='batchmean')
        ac_logits_teacher = self.teacher.get_act_logits(ob_no)
        act_logits_student = ptu.from_numpy(self.exploitation_critic.qa_values(ob_no))
        kl_div = distillation_loss(
                        F.log_softmax(act_logits_student, dim=1), 
                        F.softmax(ac_logits_teacher / self.T, dim=1)) 
        
        distill_reward = ptu.from_numpy(kl_div.data)
        print(distill_reward.shape)
        return super().train(ob_no, ac_na, distill_reward, next_ob_no, terminal_n)
    
    # Returns avg cross entropy between teacher original action logit for ob_no and teacher actions under augmented ob_no
    def get_teacher_uncertainty(self, ob_no: np.ndarray, teacher_ac_logits: torch.Tensor) -> torch.Tensor():
        #TODO: Get Data Augmentations (and apply to ob_no). Get Teacher Logits for augmented obs
        #TODO: Return avg cross entropy between o.g. action logits and new action logits
        raise NotImplementedError
        
    def get_action(self, ob):
        return self.actor.get_action(ob)