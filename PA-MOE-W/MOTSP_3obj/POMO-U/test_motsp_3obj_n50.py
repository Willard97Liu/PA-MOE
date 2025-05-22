##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
from utils.cal_pareto_demo import Pareto_sols
from utils.cal_ps_hv import cal_ps_hv

from MOTSPTester_3obj import TSPTester as Tester
from MOTSProblemDef_3obj import get_random_problems

##########################################################################################
import time
import hvwfg
import pickle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
env_params = {
    'problem_size': 50,
    'pomo_size': 50,
}

model_params = {
    'num_experts': 4,
    'topk': 2,
    'routing_level': 'node',
    'routing_method': 'input_choice',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'hyper_hidden_dim': 256,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    "dec_method": "WS",
    'model_load': {
        'path': './result/train__tsp_mix',
        'info': "MOTSP_3obj_50 Author's code (Retrain test WS)",
        'epoch': 200, # epoch version of pre-trained model to laod.
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    # 'aug_factor': 1, #512
    # 'aug_batch_size': 200,
    'aug_factor': 64,  #512
    'aug_batch_size': 20,

    'n_sols': 105  # 105, 1035, 10011
}
if tester_params['aug_factor'] > 1:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']
# if tester_params['augmentation_enable']:
#     tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n50',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)
            
def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

##########################################################################################

logger_start = time.time()

def main(n_sols = tester_params['n_sols']):
    if DEBUG_MODE:
        _set_debug_mode()

    if tester_params['aug_factor'] == 1:
        sols_floder = f"PMOCO_mean_sols_n{env_params['problem_size']}.txt"
        sols2_floder = f"PMOCO_mean_sols2_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO_hv_n{env_params['problem_size']}.txt"
    else:
        sols_floder = f"PMOCO(aug)_mean_sols_n{env_params['problem_size']}.txt"
        sols2_floder = f"PMOCO(aug)_mean_sols2_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO(aug)_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO(aug)_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO(aug)_hv_n{env_params['problem_size']}.txt"


    create_logger(**logger_params)
    _print_config()

    timer_start = time.time()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)
    

    test_path = f"./data/testdata_tsp_3o_size{env_params['problem_size']}.pt"
    shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)
    # shared_problem = get_random_problems(tester_params['test_episodes'], env_params['problem_size'])

    if n_sols == 105:
        uniform_weights = torch.Tensor(das_dennis(13, 3))  # 105
    elif n_sols == 1035:
        uniform_weights = torch.Tensor(das_dennis(44, 3))  # 1035
    elif n_sols == 10011:
        uniform_weights = torch.Tensor(das_dennis(140, 3))  # 10011
    else:
        raise NotImplementedError

    batch_size = shared_problem.shape[0]
    sols = np.zeros([batch_size, n_sols, 3])
    # hvs = np.zeros([batch_size, 1])
    mini_batch_size = tester_params['test_batch_size']
    b_cnt = tester_params['test_episodes'] / mini_batch_size
    b_cnt = int(b_cnt)
    # time_list = np.zeros((n_sols, 1))
    total_test_time = 0
    for bi in range(0, b_cnt):
        b_start = bi * mini_batch_size
        b_end = b_start + mini_batch_size
        for i in range(n_sols):
            pref = uniform_weights[i]
            # aug_score = tester.run(shared_problem,pref)
            test_timer_start = time.time()
            aug_score = tester.run(shared_problem, pref, episode=b_start)
            test_timer_end = time.time()
            # total_time = test_timer_end - test_timer_start
            # time_list[i] = total_time
            total_test_time += test_timer_end - test_timer_start
            print('Ins{:d} Test Time(s): {:.4f}'.format(i, test_timer_end - test_timer_start))

            # sols[i] = np.array(aug_score)
            sols[b_start:b_end, i, 0] = np.array(aug_score[0].flatten())
            sols[b_start:b_end, i, 1] = np.array(aug_score[1].flatten())
            sols[b_start:b_end, i, 2] = np.array(aug_score[2].flatten())

    timer_end = time.time()
    total_time = timer_end - timer_start
    # print('Avg Test Time(s): {:.4f}\n'.format(time_list.sum()))
    # print('Avg Test Time(s): {:.4f}\n'.format(total_test_time))
    # print('Run Time(s): {:.4f}\n'.format(total_time))


    max_obj1 = sols.reshape(-1, 3)[:, 0].max()
    max_obj2 = sols.reshape(-1, 3)[:, 1].max()
    max_obj3 = sols.reshape(-1, 3)[:, 2].max()
    txt2 = F"{tester.result_folder}/max_cost_n{env_params['problem_size']}.txt"
    f = open(
        txt2,
        'a')
    f.write(f"MAX OBJ1:{max_obj1}\n")
    f.write(f"MAX OBJ2:{max_obj2}\n")
    f.write(f"MAX OBJ3:{max_obj3}\n")
    f.close()


    sols_mean = sols.mean(0)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(sols_mean[:,0],sols_mean[:,1],sols_mean[:,2], marker = 'o', c = 'C1')
    
    max_lim = np.max(sols_mean, axis = 0) * 1.1
    min_lim = np.min(sols_mean, axis = 0) * 0.9
        
    ax.set_xlim(min_lim[0],max_lim[0])
    ax.set_ylim(max_lim[1],min_lim[1])
    ax.set_zlim(min_lim[2],max_lim[2])

    plt.savefig(F"{tester.result_folder}/{pareto_fig}")
    np.savetxt(F"{tester.result_folder}/{sols_floder}", sols_mean,
               delimiter='\t', fmt="%.4f\t%.4f\t%.4f")

    # ref = np.array([20,20,20])    #20
    ref = np.array([35,35,35])   #50
    #ref = np.array([65,65,65])   #100
    # ref = np.array([30, 30, 30])  # 50

    # test_timer_start = time.time()
    nd_sort = Pareto_sols(p_size=env_params['problem_size'], pop_size=sols.shape[0], obj_num=sols.shape[2])
    sols_t = torch.Tensor(sols)
    nd_sort.update_PE(objs=sols_t)
    p_sols, p_sols_num, _ = nd_sort.show_PE()
    # test_timer_end = time.time()
    # total_test_time += test_timer_end - test_timer_start
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)

    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    print('NDS: {:.4f}'.format(p_sols_num.float().mean()))
    print('Avg Test Time(s): {:.4f}\n'.format(total_test_time))

    np.savetxt(F"{tester.result_folder}/{all_sols_floder}", sols.reshape(-1, 3),
               delimiter='\t', fmt="%.4f\t%.4f\t%.4f")
    np.savetxt(F"{tester.result_folder}/{hv_floder}", hvs,
               delimiter='\t', fmt="%.4f")



    if tester_params['aug_factor'] == 1:
        f = open(
            F"{tester.result_folder}/PMOCO-TSP_3obj{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO-TSP_3obj{env_params['problem_size']}\n")
    else:
        f = open(
            F"{tester.result_folder}/PMOCO(aug)-TSP_3obj{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO(aug)-TSP_3obj{env_params['problem_size']}\n")


    f.write(f"MOTSP_3obj Type1\n")
    f.write(f"Model Path: {tester_params['model_load']['path']}\n")
    f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
    f.write(f"Hyper Hidden Dim: {model_params['hyper_hidden_dim']}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
    # f.write('Test Time(s): {:.4f}\n'.format(time_list.mean()))
    f.write('Test Time(s): {:.4f}\n'.format(total_test_time))
    f.write('Run Time(s): {:.4f}\n'.format(total_time))
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write('NDS: {:.4f}\n'.format(p_sols_num.float().mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]},{ref[2]}] \n")
    f.write(f"Info: {tester_params['model_load']['info']}\n")
    # f.write(f"{compare_type}_{optim} avg_hv:{avg_hvs} s\n")
    f.close()

##########################################################################################

if __name__ == "__main__":
    main()
