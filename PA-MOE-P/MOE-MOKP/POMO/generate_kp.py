import os
import torch
import pickle

def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 3))
    return problems

def save_mokp_test_data_as_pkl(problem_sizes, batch_size=100, seed=1234, save_dir='/data/liuw2/MOE/test_data/mokp'):
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    for size in problem_sizes:
        data = get_random_problems(batch_size=batch_size, problem_size=size)
        file_path = os.path.join(save_dir, f"mokp{size}_test_seed{seed}.pkl")  # 改成 .pkl 后缀
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)  # 使用 pickle 保存
        print(f"Saved: {file_path}, shape: {data.shape}")

if __name__ == "__main__":
    save_mokp_test_data_as_pkl(problem_sizes=[50, 100, 200])
