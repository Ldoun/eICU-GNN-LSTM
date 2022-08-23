import pandas as pd
import torch
import numpy as np
from scipy import sparse
import json
import os

def get_freqs(train_diagnoses):
    return train_diagnoses.sum()

def get_device_and_dtype():
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
    device = torch.device('cpu')
    dtype = torch.cuda.sparse.ByteTensor if device.type == 'cuda' else torch.sparse.ByteTensor
    return device, dtype

def diagnoses_scores_matrix(diagnoses, freq_adjustment, save_edge_values, batch_size, genders_scores, ages_scores, k, alpha, beta, gamma):

    freq_adjustment = 1 / freq_adjustment
    freq_adjustment = torch.tensor(freq_adjustment * 1000, device=device).type(dtype) + 1
    diagnoses_scores = torch.sparse.mm(diagnoses * freq_adjustment.unsqueeze(0), diagnoses.permute(1, 0)) # 여기가 오래 걸리넹
    no_pts = len(diagnoses)
    diags_per_pt = diagnoses.sum(axis=1)
    diags_per_pt = diags_per_pt.type(torch.ShortTensor)
    del diagnoses
    
    if save_edge_values:
        edges_val = sparse.lil_matrix((no_pts, no_pts), dtype=np.int16)
    edges = sparse.lil_matrix((no_pts, no_pts), dtype=np.uint8)  
    down = torch.split(diags_per_pt.repeat(no_pts, 1), batch_size, dim=0)
    across = torch.split(diags_per_pt.repeat(no_pts, 1).permute(1, 0), batch_size, dim=0)
    del diags_per_pt
    diagnoses_scores = diagnoses_scores.fill_diagonal_(0)
    diagnoses_score = torch.split(diagnoses_scores, batch_size, dim=0)
    del diagnoses_scores
    genders_score = torch.split(genders_scores, batch_size, dim=0)
    del genders_scores
    ages_score = torch.split(ages_scores, batch_size, dim=0)
    del ages_scores

    prev_pts = 0

    for i, (d, a, d_s, g_s, a_s) in enumerate(zip(down, across, diagnoses_score, genders_score, ages_score)):

        print('==> Processed {} patients'.format(prev_pts))
        total_combined_diags = d + a
        s_pen = 5 * d_s - total_combined_diags # s_pen을 연결해서 리턴할 수 있으면 그러면 되는데
        total_scores = int(alpha) * s_pen + int(beta) * g_s + int(gamma) * a_s # 여기 인자값을 조절해줘야 함. a_s는 작을수록 좋은거라 음수 곱해줘야 함
        
        for patient in range(len(d)):
            k_highest_inds = torch.sort(total_scores[patient].flatten()).indices[-k:]
            if save_edge_values:
                k_highest_vals = torch.sort(total_scores[patient].flatten()).values[-k:]
                for i, val in zip(k_highest_inds, k_highest_vals):
                    if val == 0:
                        val = 1
                    edges_val[patient + prev_pts, i] = val
            for i in k_highest_inds:
                edges[patient + prev_pts, i] = 1
        prev_pts += batch_size
        
    del diagnoses_score, genders_score, ages_score, down, across, d, a, d_s, a_s, g_s, total_combined_diags, s_pen, total_scores

    edges = edges + edges.transpose()
    if save_edge_values:
        edges_val = edges_val + edges_val.transpose()
        for i, (edge, edge_val) in enumerate(zip(edges, edges_val)):

            edges_val[i, edge.indices] = edge_val.data // edge.data
        edges = edges_val
    edges.setdiag(0)
    edges.eliminate_zeros()

    edges = sparse.tril(edges, k=-1) # -> 대각선 위 없애서 중복 막아주는 애

    v, u, vals = sparse.find(edges)
    del edges
    
    return u, v, vals, k

def genders_scores_matrix(genders): # and 연산 할까
    genders_scores = torch.bitwise_and(genders, genders.permute(1, 0))
    del genders
    
    return genders_scores

def ages_scores_matrix(ages): # 빼고 절댓값    
    ages_scores = torch.sub(ages, ages.permute(1, 0))
    del ages
    ages_scores = torch.abs(ages_scores)

    
    return ages_scores

    
# if __name__ == '__main__':
def get_graph(config):
    device, dtype = get_device_and_dtype()
    
    with open('paths.json', 'r') as f:
        loader = json.load(f)

        eICU_path = loader["eICU_path"]
        graph_dir = loader["graph_dir"]
    
    adjust = '_adjusted' 
    
    if not os.path.exists(graph_dir):  # make sure the graphs folder exists
        os.makedirs(graph_dir)

    train_flat = pd.read_csv('{}train/flat.csv'.format(eICU_path), index_col='patient')
    val_flat = pd.read_csv('{}val/flat.csv'.format(eICU_path), index_col='patient')
    test_flat = pd.read_csv('{}test/flat.csv'.format(eICU_path), index_col='patient')

    train_gender = train_flat[['gender']]
    val_gender = val_flat[['gender']]
    test_gender = test_flat[['gender']]
    all_genders = pd.concat([train_gender, val_gender, test_gender], sort=False)

    del train_gender, val_gender, test_gender

    genders = np.array(all_genders).astype(np.uint8)
    del all_genders
    genders = torch.tensor(genders, device=device)
    genders_scores = genders_scores_matrix(genders)
    del genders
    
    train_age = train_flat[['age']]
    val_age = val_flat[['age']]
    test_age = test_flat[['age']]
    del train_flat, val_flat, test_flat
    all_ages = pd.concat([train_age, val_age, test_age], sort=False)
    del train_age, val_age, test_age
    ages = np.array(all_ages).astype(np.uint8)
    del all_ages
    ages = torch.tensor(ages, device=device)
    ages_scores = ages_scores_matrix(ages)
    del ages
    
    train_diagnoses = pd.read_csv('{}train/diagnoses.csv'.format(eICU_path), index_col='patient')
    val_diagnoses = pd.read_csv('{}val/diagnoses.csv'.format(eICU_path), index_col='patient')
    test_diagnoses = pd.read_csv('{}test/diagnoses.csv'.format(eICU_path), index_col='patient')
    all_diagnoses = pd.concat([train_diagnoses, val_diagnoses, test_diagnoses], sort=False)

    freq_adjustment = get_freqs(train_diagnoses)
    del train_diagnoses, val_diagnoses, test_diagnoses
    diagnoses = np.array(all_diagnoses).astype(np.uint8)
    del all_diagnoses
    diagnoses = torch.tensor(diagnoses, device=device)
    # get diagnoses score matrix
    u, v, vals, k = diagnoses_scores_matrix(diagnoses, freq_adjustment, False, 1000, genders_scores, ages_scores, 3, config['alpha'], config['beta'], config['gamma'])
    del diagnoses
    
    np.savetxt('{}{}_{}_u_k={}{}.txt'.format(graph_dir, config['tuning_version'], 'k_closest', k, adjust), u.astype(int), fmt='%i')
    np.savetxt('{}{}_{}_v_k={}{}.txt'.format(graph_dir, config['tuning_version'], 'k_closest', k, adjust), v.astype(int), fmt='%i')
    #np.savetxt('{}{}_scores_k={}{}.txt'.format(graph_dir, args.mode, k, adjust), vals.astype(int), fmt='%i')
