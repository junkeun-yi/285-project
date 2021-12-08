from cs285.infrastructure.utils import *
from cs285.policies.teacher_policy import DistillationTeacherPolicy
from cs285.policies.CNN_policy import CNNPolicyDistillationStudent
from cs285.exploration.random_feat import RandomFeatCuriosity
from cs285.exploration.icm_model import ICMModel
from cs285.exploration.rnd_model import RNDModel
from .dqn_agent import DQNAgent
import cs285.infrastructure.pytorch_util as ptu
import torch
import numpy as np
from torch.nn import functional as F
from torch.nn import KLDivLoss
from torchvision import transforms

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
        self.T = self.agent_params['temperature']

        # set curiosity model (if using)
        self.curiosity = self.agent_params["use_curiosity"]
        if self.curiosity:
            print("Using RND")
            self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
            self.curiosity_weight = self.agent_params['explore_weight_schedule']
        #     if self.agent_params["use_icm"]:
        #         print("Using ICM Model")
        #         self.exploration_model = ICMModel(agent_params, self.optimizer_spec)
        #     else:
        #         print("Using Random Features")
        #         self.exploration_model = RandomFeatCuriosity(agent_params, self.optimizer_spec)
        
        # Flag if using uncertainty weighted distillation
        self.uncertainity_aware = self.agent_params["use_uncertainty"]
        if self.uncertainity_aware:
            print("Using Uncertainty!")
            self.to_img = transforms.ToPILImage()
            self.data_aug = [
                                transforms.ColorJitter(brightness=0.05),
                                transforms.Compose([
                                                        transforms.RandomAffine(10),
                                                        transforms.Resize(self.agent_params['ob_dim'][:-1]) #ob_dim is (H,W,C) need (H,W)            
                                ]),
                                transforms.Compose([
                                                        transforms.Pad(2),
                                                        transforms.Resize(self.agent_params['ob_dim'][:-1])
                                ])
            ]

        self.eval_policy = self.actor

    def train(self, ob_no: np.ndarray, ac_na: np.ndarray, re_n: np.ndarray, next_ob_no: np.ndarray, terminal_n: np.ndarray):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            obs = ptu.from_numpy(ob_no)

            # retrieve teacher's action logits on observations
            ac_logits_teacher = self.teacher.get_act_logits(ob_no)
            ac_logits_student = self.actor(obs).logits
            acs = F.gumbel_softmax(ac_logits_student, hard=True, dim=1)

            # Get KL Loss
            kl_loss = KLDivLoss(reduction='batchmean')
            loss = kl_loss(
                F.log_softmax(ac_logits_student, dim=1), 
                F.softmax(ac_logits_teacher / self.T, dim=1))

            # If using uncertainty, find teacher uncertainty under data augmentations
            uncertainity_weight = None
            if self.uncertainity_aware:
                uncertainity_weight = self.get_teacher_uncertainty(ob_no, ac_logits_teacher)
                log["Teacher Average Uncertainty"] = uncertainity_weight

            if self.curiosity:
                intrinsic_rew = torch.mean(self.exploration_model(obs, acs))

                log["Intrinsic Reward"] = intrinsic_rew
                
                curiosity_weight = self.curiosity_weight.value(self.t)
                if uncertainity_weight is not None:
                    curiosity_weight = uncertainity_weight

                loss -= curiosity_weight * intrinsic_rew


            self.actor.update(loss)
            log['Actor Loss'] = loss.item() #Loss term now includes intrinsic_rew if curiosity used

            if self.curiosity:
                curiosity_loss = self.exploration_model.update(ob_no, acs.detach())
                log['RND Loss'] = curiosity_loss

            self.num_param_updates += 1

        self.t += 1
        return log
    

    # Returns avg cross entropy between teacher original action logit for ob_no and teacher actions under augmented ob_no
    def get_teacher_uncertainty(self, ob_no: np.ndarray, teacher_ac_logits: torch.Tensor) -> torch.Tensor():
        
        teacher_ac_probs = F.softmax(teacher_ac_logits, dim=1)
        obs = ptu.from_numpy(ob_no).permute((0,3,1,2))
        uncertainty = 0
        for data_aug in self.data_aug:
            obs_prime = data_aug(obs).permute((0,2,3,1))
            np_obs_prime = ptu.to_numpy(obs_prime)
            teacher_aug_logit = self.teacher.get_act_logits(np_obs_prime)
            log_prob_aug = F.log_softmax(teacher_aug_logit)
            if uncertainty is None:
                uncertainty = (-teacher_ac_probs * log_prob_aug).sum()
            else:
                uncertainty += (-teacher_ac_probs * log_prob_aug).sum()

        return uncertainty.cpu().item() / (ob_no.shape[0] * len(self.data_aug))
        
        
    def get_action(self, ob):
        return self.actor.get_action(ob)
