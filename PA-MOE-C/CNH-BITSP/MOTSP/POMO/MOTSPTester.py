import torch

import os
from logging import getLogger

from MOTSPEnv import TSPEnv as Env
from models.MOTSPModel import TSPModel as Model

from MOTSProblemDef import augment_preference, augment_32_preference, augment_xy_data_by_64_fold_2obj, augment_xy_data_by_32_fold_2obj


from einops import rearrange

from utils.utils import *


class TSPTester:
    def __init__(self,
                 model_params,
                 tester_params):

        # save arguments
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

    def run(self, shared_problem, pref):
        self.time_estimator.reset()
    
        aug_score_AM = {}
        
        # 2 objs
        for i in range(2):
            aug_score_AM[i] = AverageMeter()
            
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            aug_score = self._test_one_batch(shared_problem, pref, batch_size, episode)
            
            # 2 objs
            for i in range(2):
                aug_score_AM[i].update(aug_score[i], batch_size)

            episode += batch_size
           

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            all_done = (episode == test_num_episode)
            if all_done:
                self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f} ".format(aug_score_AM[0].avg, aug_score_AM[1].avg))
                
        return [aug_score_AM[0].avg.cpu(), aug_score_AM[1].avg.cpu()]
                
    def _test_one_batch(self, shared_probelm, pref, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        _, problem_size, _ = shared_probelm.size()
            
        self.env.problem_size = problem_size
        self.env.pomo_size = problem_size
        self.env.batch_size = batch_size
        self.env.instances= shared_probelm[episode: episode + batch_size]
        self.env.preference = pref[episode: episode + batch_size]
        
        if aug_factor == 64:
            self.env.batch_size = self.env.batch_size * 64
            self.env.instances = augment_xy_data_by_64_fold_2obj(self.env.instances)
            self.env.preference = augment_preference(self.env.preference)
            # self.env.preference = self.env.preference.repeat(aug_factor, 1)
            
        if aug_factor == 32:
            self.env.batch_size = self.env.batch_size * 32
            self.env.instances = augment_xy_data_by_32_fold_2obj(self.env.instances)
            self.env.preference = augment_32_preference(self.env.preference)
            
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)
        
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            instances = reset_state.instances
            pref = reset_state.preference
            self.model.pre_forward(instances, pref)
            
        state, reward, done = self.env.pre_step()
        
        while not done:
            
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            

        
        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward
        z = torch.ones(reward.shape).to(reward.device) * 0.0
        a = pref[:, 0]
        b = pref[:, 1]
        x = 1 / (1 + b / a)
        y = 1 - x
        preference = torch.cat((x[:, None], y[:, None]), dim=-1)
        new_pref = preference[:, None, :].expand_as(reward)
        tch_reward = new_pref * (reward - z)
        tch_reward, _ = tch_reward.max(dim=2)

        reward = - reward
        tch_reward = -tch_reward

        tch_reward = tch_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        
        tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)') 
        _ , max_idx_aug = tch_reward_aug.max(dim=1)
        max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0],1)
        max_reward_obj1 = rearrange(reward[:,:,0].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
        max_reward_obj2 = rearrange(reward[:,:,1].reshape(aug_factor, batch_size, self.env.pomo_size), 'c b h -> b (c h)').gather(1, max_idx_aug)
     
        aug_score = []
        aug_score.append(-max_reward_obj1.float().mean())
        aug_score.append(-max_reward_obj2.float().mean())

        # aug_score.append(-max_reward_obj1.float())
        # aug_score.append(-max_reward_obj2.float())
        
        return aug_score

     
