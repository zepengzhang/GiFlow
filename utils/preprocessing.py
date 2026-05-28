import torch
import scipy
import scipy.sparse as sp
import numpy as np
import torch

def operator_generation(L, k_eig=50, t=0.3798):
    if k_eig >= L.shape[0]:
        L_dense = L.toarray()
        evals, evecs = np.linalg.eigh(L_dense)
    else:
        evals, evecs = scipy.sparse.linalg.eigsh(L, k=k_eig)
    Phi = torch.tensor(evecs, dtype=torch.float32)  
    Lambda = torch.tensor(evals, dtype=torch.float32) 

    exp_tL = torch.diag(torch.exp(-t * Lambda))                
    H = Phi @ exp_tL @ Phi.T                                

    return H

def operator_generation_decompose(L, k_eig=50, t=1):
    if k_eig >= L.shape[0]:
        L_dense = L.toarray()
        evals, evecs = np.linalg.eigh(L_dense)
    else:
        evals, evecs = scipy.sparse.linalg.eigsh(L, k=k_eig)
    Phi = torch.tensor(evecs, dtype=torch.float32) 
    Lambda = torch.tensor(evals, dtype=torch.float32)  

    return Phi, Lambda

def generate_time_laplacian(T):
    A = sp.lil_matrix((T, T))

    for i in range(1, T - 1):
        A[i, i - 1] = 1  
        A[i, i + 1] = 1 

    A[0, 1] = 1 
    A[T - 1, T - 2] = 1 

    A = A.tocsr()
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    I = sp.eye(T)
    L_sym = I - D_inv_sqrt @ A @ D_inv_sqrt

    return L_sym


def graph_diffusion(x, H_t):
    B, N, T, F = x.shape
    x = x.permute(0, 2, 3, 1) 
    x = x.reshape(B * T * F, N)
    x_diffused = H_t @ x.T
    x_diffused = x_diffused.T.view(B, T, F, N)
    x_diffused = x_diffused.permute(0, 3, 1, 2)

    return x_diffused

def time_diffusion(x, H_t):
    B, N, T, F = x.shape 
    x = x.permute(0, 1, 3, 2) 
    x = x.reshape(B * N * F, T)
    x_diffused = x @ H_t.T
    x_diffused = x_diffused.view(B, N, F, T)
    x_diffused = x_diffused.permute(0, 1, 3, 2)

    return x_diffused

def preprocess_fm(batch, H_spatial, H_temporal, window):

    x = (batch.input.x).transpose(1, 2)    
    eval_mask = (batch.eval_mask).transpose(1, 2)    
    x = torch.where(eval_mask, torch.zeros_like(x), x)
    
    if H_spatial is not None or H_temporal is not None:
        if H_spatial is not None:
            x_diff = graph_diffusion(x, H_spatial)
        if H_temporal is not None:
            x_diff = time_diffusion(x, H_temporal)
        x = torch.where(eval_mask, x_diff, x)

    y = (batch.target.y).transpose(1, 2)
    y = batch.transform['y'].transform(y)
    u = batch.input.u
    
    t = torch.rand(x.shape[0]).to(x.device) 
    t_expanded = t.view(-1, 1, 1).expand(-1, window, -1)  
    u_expand = torch.cat([u, t_expanded], dim=-1)
    t = t.view(-1, 1, 1, 1)
    x_t = (1 - t) * x + t * y
    v_t = y - x
    
    return x, x_t, v_t, u_expand, eval_mask, t

def preprocess_fm_test(batch, H_spatial, H_temporal):

    x = (batch.input.x).transpose(1, 2)   
    eval_mask = (batch.eval_mask).transpose(1, 2)    
    x = torch.where(eval_mask, torch.zeros_like(x), x)

    if H_spatial is not None or H_temporal is not None:
        if H_spatial is not None:
            x_diff = graph_diffusion(x, H_spatial)
        if H_temporal is not None:
            x_diff = time_diffusion(x, H_temporal)
        x = torch.where(eval_mask, x_diff, x)

    y = (batch.target.y).transpose(1, 2)
    u = batch.input.u

    return x, x, y, u, eval_mask