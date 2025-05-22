##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 1
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

from MOCVRPTester import CVRPTester as Tester
from MOCVRProblemDef import get_random_problems
##########################################################################################
import time
import hvwfg


from matplotlib import pyplot as plt
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

info = "Retrained MOCVRP50 for 200 epochs"
tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'dec_method': 'WS',
    'model_load': {
        'path': './result/train_cvrp_mix',
        'info': "MOCVRP50 Retrain WS Test WS Right Makespan",
        'epoch': 200,
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 4, #8,
    # 'aug_factor': 8, #8,
    'aug_batch_size': 100
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__cvrp_n50',
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
##########################################################################################
def main(n_sols = 101):

    if tester_params['aug_factor'] == 1:
        sols_floder = "PMOCO_mean_sols.txt"
        pareto_fig = "PMOCO_Pareto.png"
        all_sols_floder = "PMOCO_all_mean_sols.txt"
        hv_floder = "PMOCO_hv.txt"
    else:
        sols_floder = "PMOCO(aug)_mean_sols.txt"
        pareto_fig = "PMOCO(aug)_Pareto.png"
        all_sols_floder = "PMOCO(aug)_all_mean_sols.txt"
        hv_floder = "PMOCO(aug)_hv.txt"

    logger_start = time.time()

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    timer_start = time.time()
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    test_path = f"./data/testdata_cvrp_size{env_params['problem_size']}.pt"
    # test_path = f"./data/eil_50.pt"
    data = torch.load(test_path)
    shared_node_demand = data['demand_data'].squeeze(-1).to(device=CUDA_DEVICE_NUM)
    shared_depot_xy = data['node_data'][:, 0, :].unsqueeze(1).to(device=CUDA_DEVICE_NUM)
    shared_node_xy = data['node_data'][:, 1:, :].to(device=CUDA_DEVICE_NUM)
    # print(shared_problem)
    # shared_depot_xy, shared_node_xy, shared_node_demand = get_random_problems(tester_params['test_episodes'], env_params['problem_size'])


    # right makespan
    # ref = np.array([30,4])   #20
    ref = np.array([45,4])   #50
    # ref = np.array([80,4])   #100
    # ref = np.array([40, 3])  # 50

    batch_size = shared_node_xy.shape[0]
    sols = np.zeros([batch_size, n_sols, 2])
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
            pref = torch.zeros(2).cuda()
            pref[0] = 1 - i / (n_sols - 1)
            pref[1] = i / (n_sols - 1)
            pref = pref / torch.sum(pref)

            test_timer_start = time.time()
            aug_score = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref, episode=b_start)
            test_timer_end = time.time()
            total_test_time += test_timer_end - test_timer_start
            print('Ins{:d} Test Time(s): {:.4f}'.format(i, test_timer_end - test_timer_start))

            sols[b_start:b_end, i, 0] = np.array(aug_score[0].flatten())
            sols[b_start:b_end, i, 1] = np.array(aug_score[1].flatten())


    timer_end = time.time()
    total_time = timer_end - timer_start
    # print('Avg Test Time(s): {:.4f}\n'.format(time_list.sum()))

    max_obj1 = sols.reshape(-1, 2)[:, 0].max()
    max_obj2 = sols.reshape(-1, 2)[:, 1].max()
    txt2 = F"{tester.result_folder}/max_cost_n{env_params['problem_size']}.txt"
    f = open(
        txt2,
        'a')
    f.write(f"MOCVRP{env_params['problem_size']}\n")
    f.write(f"MAX OBJ1:{max_obj1}\n")
    f.write(f"MAX OBJ2:{max_obj2}\n")
    f.close()

    fig = plt.figure()


    sols_mean = sols.mean(0)
    plt.plot(sols_mean[:,0],sols_mean[:,1], marker = 'o', c = 'C1',ms = 3,  label='PSL-MOCO (Ours)')
    plt.savefig(F"{tester.result_folder}/{pareto_fig}")
    # np.savetxt(F"{tester.result_folder}/PMOCO_mean_sols.txt", sols.reshape(-1, 2),
    #            delimiter='\t', fmt="%.4f\t%.4f")

    np.savetxt(F"{tester.result_folder}/{sols_floder}", sols.mean(0),
               delimiter='\t', fmt="%.4f\t%.4f")

    plt.legend()

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

    np.savetxt(F"{tester.result_folder}/{all_sols_floder}", sols.reshape(-1, 2),
               delimiter='\t', fmt="%.4f\t%.4f")
    np.savetxt(F"{tester.result_folder}/{hv_floder}", hvs,
               delimiter='\t', fmt="%.4f")

    if tester_params['aug_factor'] == 1:
        f = open(
            F"{tester.result_folder}/PMOCO-CVRP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO-CVRP{env_params['problem_size']}\n")
    else:
        f = open(
            F"{tester.result_folder}/PMOCO(aug)-CVRP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO(aug)-CVRP{env_params['problem_size']}\n")

    f.write(f"Model Path: {tester_params['model_load']['path']}\n")
    f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
    f.write(f"Hyper Hidden Dim: {model_params['hyper_hidden_dim']}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
    f.write('Test Time(s): {:.4f}\n'.format(total_test_time))
    f.write('Run Time(s): {:.4f}\n'.format(total_time))
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write('NDS: {:.4f}\n'.format(p_sols_num.float().mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]}] \n")
    f.write(f"Info: {tester_params['model_load']['info']}\n")
    # f.write(f"{compare_type}_{optim} avg_hv:{avg_hvs} s\n")
    f.close()


##########################################################################################
if __name__ == "__main__":
    main()