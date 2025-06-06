import torch

import os
from logging import getLogger

from MOTSPEnv_3obj import TSPEnv as Env
from models.MOTSPModel import TSPModel as Model

from MOTSProblemDef_3obj import get_random_problems, augment_xy_data_by_n_fold_3obj

from einops import rearrange

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env()
        self.model = Model(**self.model_params)
        
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, shared_problem, pref, episode=0):
        self.time_estimator.reset()
        
        aug_score_AM = {}
        
        # 3 objs
        for i in range(3):
            aug_score_AM[i] = AverageMeter()
            
        test_num_episode = self.tester_params['test_episodes']
        # episode = 0
        episode = episode

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            aug_score = self._test_one_batch(shared_problem, pref, batch_size, episode)
            
            # 3 objs
            for i in range(3):
                aug_score_AM[i].update(aug_score[i], batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            all_done = (episode == test_num_episode)
            self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f}, AUG_OBJ_3 SCORE: {:.4f} ".format(aug_score_AM[0].avg.mean(), aug_score_AM[1].avg.mean(), aug_score_AM[2].avg.mean()))
            break
        return [aug_score_AM[0].avg.cpu(), aug_score_AM[1].avg.cpu(), aug_score_AM[2].avg.cpu()]
              
    def _test_one_batch(self, shared_probelm, pref, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        self.env.problem_size = self.env_params['problem_size']
        self.env.pomo_size = self.env_params['pomo_size']
        self.env.batch_size = batch_size
        self.env.problems = shared_probelm[episode: episode + batch_size]
        
        if aug_factor > 1:
            self.env.batch_size = self.env.batch_size * aug_factor
            self.env.problems = augment_xy_data_by_n_fold_3obj(self.env.problems, aug_factor)
            
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)
      
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            
            # self.model.decoder.assign(pref)
            self.model.pre_forward(reset_state, pref)
            
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            
        # if self.env_params['problem_size'] == 20:
        #     ref_values = 15
        # if self.env_params['problem_size'] == 50:
        #     ref_values = 30
        # if self.env_params['problem_size'] == 100:
        #     ref_values = 45
        
        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward

        if self.tester_params['dec_method'] == "WS":
            tch_reward = (pref * reward).sum(dim=2)
            tch_reward = - tch_reward
        elif self.tester_params['dec_method'] == "TCH":
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward, _ = tch_reward.max(dim=2)
        elif self.tester_params['dec_method'] == "PBI":
            if self.env_params['problem_size'] == 20:
                ref_values = 15
            if self.env_params['problem_size'] == 50:
                ref_values = 30
            if self.env_params['problem_size'] == 100:
                ref_values = 45
            z = torch.ones(reward.shape).cuda() * ref_values
            theta = 0.1
            d1 = (pref * (z - reward)).sum(dim=2) / torch.norm(pref)
            d1_variant = pref[:, None, None] / torch.norm(pref) * d1
            d1_varaint = rearrange(d1_variant, 'c b h -> b h c')
            d2 = torch.norm(z - reward - d1_varaint)
            pbi_reward = d1 - theta * d2
            tch_reward = pbi_reward
        else:
            return NotImplementedError

        # z = torch.ones(reward.shape).cuda() * ref_values
        #
        # theta = 0.1
        #
        # d1 = (pref * (z - reward)).sum(dim=2) / torch.norm(pref)
        # d1_variant = pref[:,None,None] / torch.norm(pref) * d1
        # d1_varaint = rearrange(d1_variant, 'c b h -> b h c')
        # d2 = torch.norm(z - reward - d1_varaint)
        # pbi_reward = d1 - theta * d2
        # tch_reward = pbi_reward
        
        reward = - reward
        
        tch_reward = tch_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        
        tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)') 
        _ , max_idx_aug = tch_reward_aug.max(dim=1)
        max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0],1)
        max_reward_obj1 = rearrange(reward[:, :, 0].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
        max_reward_obj2 = rearrange(reward[:, :, 1].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
        max_reward_obj3 = rearrange(reward[:, :, 2].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
     
        aug_score = []
        
        # aug_score.append(-max_reward_obj1.float().mean())
        # aug_score.append(-max_reward_obj2.float().mean())
        # aug_score.append(-max_reward_obj3.float().mean())
        aug_score.append(-max_reward_obj1.float())
        aug_score.append(-max_reward_obj2.float())
        aug_score.append(-max_reward_obj3.float())
        
        return aug_score

       