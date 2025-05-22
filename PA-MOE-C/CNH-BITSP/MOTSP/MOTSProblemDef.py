import torch

def get_training_problems(batch_size, problem_size):
    instances = torch.rand(size=(batch_size, problem_size, 4))
    # pref = torch.rand([2])
    # pref = pref / torch.sum(pref)
    # preference = pref[None, :].expand(batch_size, 2)
    preference = torch.Tensor(batch_size, 2).uniform_(1e-6, 1)
    problems = {
        'instances': instances,
        'preference': preference
    }
    return problems


def get_random_problems(batch_size, problem_size):
    instances = torch.rand(size=(batch_size, problem_size, 4))
    preference = torch.Tensor(batch_size, 2).uniform_(1e-6, 1)
    problems = {
        'instances': instances,
        'preference': preference
    }
    return problems

def augment_xy_data_by_64_fold_2obj(xy_data):
   
    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]

    dat1 = {}
    dat2 = {}
    
    dat_aug = []
    
    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1]= torch.cat((1-x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1-y1), dim=2)
    dat1[3] = torch.cat((1-x1, 1-y1), dim=2)
    dat1[4]= torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1-y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1-x1), dim=2)
    dat1[7] = torch.cat((1-y1, 1-x1), dim=2)
    
    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1]= torch.cat((1-x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1-y2), dim=2)
    dat2[3] = torch.cat((1-x2, 1-y2), dim=2)
    dat2[4]= torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1-y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1-x2), dim=2)
    dat2[7] = torch.cat((1-y2, 1-x2), dim=2)
    
    for i in range(8):
        for j in range(8):
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)
            
    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems


def augment_preference(pref):
    coff = torch.Tensor(63).uniform_(1e-6, 1)
    new_pref = []
    new_pref.append(pref)
    for i in range(len(coff)):
        new_pref.append(coff[i] * pref)

    return torch.cat(new_pref, dim=0)


def augment_32_preference(pref):
    coff = torch.Tensor(31).uniform_(1e-6, 1)
    new_pref = []
    new_pref.append(pref)
    for i in range(len(coff)):
        new_pref.append(coff[i] * pref)

    return torch.cat(new_pref, dim=0)


def augment_xy_data_by_32_fold_2obj(xy_data):
    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]

    dat1 = {}
    dat2 = {}
    dat_aug = []

    for i in range(8):
        dat1[i] = torch.cat((x1 if i < 4 else y1, 
                             y1 if i < 4 else x1), dim=2)
        if i % 2 == 1:
            dat1[i] = torch.cat((1 - dat1[i][:, :, [0]], dat1[i][:, :, [1]]), dim=2)
        if (i // 2) % 2 == 1:
            dat1[i] = torch.cat((dat1[i][:, :, [0]], 1 - dat1[i][:, :, [1]]), dim=2)

    for j in range(8):
        dat2[j] = torch.cat((x2 if j < 4 else y2, 
                             y2 if j < 4 else x2), dim=2)
        if j % 2 == 1:
            dat2[j] = torch.cat((1 - dat2[j][:, :, [0]], dat2[j][:, :, [1]]), dim=2)
        if (j // 2) % 2 == 1:
            dat2[j] = torch.cat((dat2[j][:, :, [0]], 1 - dat2[j][:, :, [1]]), dim=2)

    count = 0
    for i in range(8):
        for j in range(8):
            if count >= 32:
                break
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)
            count += 1
        if count >= 32:
            break

    aug_problems = torch.cat(dat_aug, dim=0)
    return aug_problems