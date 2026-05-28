import os
import tqdm
import wandb
import torch
import sys
from contextlib import redirect_stdout, redirect_stderr
import torch.nn as nn
from ptflops import get_model_complexity_info
import time
import random
import numpy as np
from argparse import ArgumentParser
from data import dataset_loading
import models
import pytorch_lightning as pl
from utils import *
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--dataset_name', default='airquality_small', type=str, choices=['AirQuality','airquality_small', 'PeMS04', 'PeMS08','synthetic'])
    parser.add_argument('--model_state', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--missing_rate', default=0.2, type=float)
    parser.add_argument('--missing_type', default='point', type=str)

    parser.add_argument('--model_name', default='FlowMatching', type=str, choices=['FlowMatching'])
    parser.add_argument('--diffusion_type', default='graph_time', type=str, choices=['graph', 'time', 'graph_time', 'none'])

    parser.add_argument('--seed', default=393, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)

    parser.add_argument('--k_eig', type=int, default=50)
    parser.add_argument('--tau_s', type=float, default=0.1425)
    parser.add_argument('--tau_t', type=float, default=0.3931)

    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--propagation_layers', type=int, default=5)
    parser.add_argument('--spatial_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--training_epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--patience', default=40, type=int, help="patience for early stopping")
    parser.add_argument('--lr_decay', default=True, type=bool)
    parser.add_argument('--lr_decay_factor', default=0.3, type=float)
    parser.add_argument('--lr_decay_patience', default=10, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--ema_start_epoch', default=30, type=int)

    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', type=str, default="./save") 
    parser.add_argument('--window', default=24, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--adj_threshold', default=0.1, type=float)
    parser.add_argument('--val_len', default=0.1, type=float)
    parser.add_argument('--test_len', default=0.2, type=float)    

    args = parser.parse_args()

    return args

def run_experiment(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    print(torch.cuda.is_available())
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    save_path = f'{args.save_dir}/{args.dataset_name}/{args.model_name}/{args.diffusion_type}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    start_time = time.time()
    dm, edge_index, edge_weight, n_nodes, time_dim = dataset_loading(dataset_name=args.dataset_name, 
                                                                                missing_rate=args.missing_rate,
                                                                                missing_type=args.missing_type,
                                                                                window=args.window,
                                                                                stride=args.stride,
                                                                                adj_threshold=args.adj_threshold,
                                                                                val_len=args.val_len,
                                                                                test_len=args.test_len,
                                                                                seed=args.seed,
                                                                                batch_size=args.batch_size)
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time used for data loading: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    start_time = end_time
    node_embed = torch.arange(n_nodes).to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    if args.diffusion_type in ['graph', 'graph_time']:
        lap_edge_index, lap_edge_weight = get_laplacian(edge_index, normalization='sym')
        L = to_scipy_sparse_matrix(lap_edge_index, lap_edge_weight, num_nodes=n_nodes).astype('float64')
        H_spatial = operator_generation(L, args.k_eig, args.tau_s)
        H_spatial = H_spatial.to(device)
    else:
        H_spatial = None
    if args.diffusion_type in ['time', 'graph_time']:
        L = generate_time_laplacian(args.window)
        H_temporal = operator_generation(L, args.k_eig, args.tau_t)
        H_temporal = H_temporal.to(device)
    else:
        H_temporal = None
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time used for diffusion matrix: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    args.time_dim = time_dim
    args.n_nodes = n_nodes

    ModelClass = getattr(models, args.model_name, None)
    if ModelClass is None:
        raise ValueError(f"Model class '{args.model_name}' not found in models.")

    model = ModelClass(args)
    model = model.to(device)

    ema = EMA(model, decay=0.9999)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=1e-5
        )
    else:
        scheduler = None
    
    early_stopping = EarlyStopping(patience=args.patience, 
                                   verbose=True, 
                                   path=f"{save_path}{args.dataset_name}_{args.model_name}_{args.missing_rate}_{args.seed}_{args.lr}_{args.dropout}_{args.hidden_dim}_{args.adj_threshold}.pt")

    if args.model_state == "train":
        start_time = time.time()
        # Load data and preprocess
        train_loader = dm.train_dataloader(batch_size=args.batch_size, shuffle=False)
        val_loader = dm.val_dataloader(batch_size=args.batch_size, shuffle=False)
        test_loader = dm.test_dataloader(batch_size=args.batch_size, shuffle=False)
        preprocessed_train_batches = []
        for batch in train_loader:
            batch = batch.to(device)
            x0, x, y, u, eval_mask, t = preprocess_fm(batch, H_spatial, H_temporal, args.window)
            preprocessed_train_batches.append((batch, x0, x, y, u, eval_mask, t))
        preprocessed_val_batches = []
        for batch in val_loader:
            batch = batch.to(device)
            x0, x, y, u, eval_mask, t = preprocess_fm(batch, H_spatial, H_temporal, args.window)
            preprocessed_val_batches.append((batch, x0, x, y, u, eval_mask, t))
        for batch in test_loader:
            batch = batch.to(device)
            x0, x, y, u, eval_mask, t = preprocess_fm(batch, H_spatial, H_temporal, args.window)
            preprocessed_val_batches.append((batch, x0, x, y, u, eval_mask, t))
        end_time = time.time()
        elapsed = end_time - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time used for preprocessing: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        start_time = end_time
        for epoch in range(args.training_epoch):
            model.train()
            optimizer.zero_grad() 
            losses = []
            for batch, x0, x, y, u, eval_mask, t in preprocessed_train_batches:
                x_rec = model(node_embed=node_embed, x=x, x0=x0, ex=u, edge_index=edge_index, mask=eval_mask)
                x_1 = x + (1 - t) * x_rec
                loss = torch.mean(torch.abs(x_rec[eval_mask] - y[eval_mask])) 

                loss.backward()
                optimizer.step()
                if epoch >= args.ema_start_epoch:
                    ema.register()
                    ema.update()
                optimizer.zero_grad()
                losses.append(loss.cpu().item())

            print(f"train_loss={np.mean(losses)} after epoch {epoch}")

            model.eval()
            if epoch >= args.ema_start_epoch:
                ema.apply_shadow()
            losses = []
            with torch.no_grad():
                for batch, x0, x, y, u, eval_mask, t in preprocessed_val_batches:
                    x_rec = model(node_embed=node_embed, x=x, x0=x0, ex=u, edge_index=edge_index, mask=eval_mask)                        
                    x_1 = x + (1 - t) * x_rec
                    loss = torch.mean(torch.abs(x_rec[eval_mask] - y[eval_mask]))

                    losses.append(loss.cpu().item())

                print(f"val_loss={np.mean(losses)} after epoch {epoch}")
                early_stopping(np.mean(losses), model)
                if epoch >= args.ema_start_epoch:
                    ema.restore() 
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                if scheduler is not None:
                    scheduler.step(np.mean(losses))

        end_time = time.time()
        elapsed = end_time - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time used for training: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    model.eval()
    ema.apply_shadow()

    losses_mae = []
    losses_mse = []
    losses_mape = []
    with torch.no_grad():
        for batch in test_loader:

            batch = batch.to(device)
            x0, x, y, u, eval_mask = preprocess_fm_test(batch, H_spatial, H_temporal)

            t = torch.zeros(x.shape[0], 1).to(x.device)
            num_steps = 20
            delta_t = 1 / num_steps

            for _ in range(num_steps):
                t_expanded = t.view(-1, 1, 1).expand(-1, args.window, -1)  # [64, 24, 1]
                u_expand = torch.cat([u, t_expanded], dim=-1)
                vt = model(node_embed=node_embed, x=x, x0=x0, ex=u_expand, edge_index=edge_index, mask=eval_mask)
                x_rec = x + vt * delta_t  
                t += delta_t
                x = torch.where(eval_mask, x_rec, x)

            x_rec = x  

            x_rec = batch.transform['y'].inverse_transform(x_rec)

            loss_mae = torch.mean(torch.abs(x_rec[eval_mask] - y[eval_mask]))
            loss_mse = torch.mean((x_rec[eval_mask] - y[eval_mask]) ** 2)
            nonzero_mask = (torch.abs(y[eval_mask]) > 1e-6)
            loss_mape = torch.mean(torch.abs((x_rec[eval_mask][nonzero_mask] - y[eval_mask][nonzero_mask]) / y[eval_mask][nonzero_mask])) * 100

            losses_mae.append(loss_mae.cpu().item())
            losses_mse.append(loss_mse.cpu().item())
            losses_mape.append(loss_mape.cpu().item())
            
        ema.restore()
        print("######################## Testing score ########################")
        print(f"{args.model_name}: {args.dataset_name}-{args.diffusion_type}-{args.missing_type}-{args.missing_rate}-{args.seed}")
        print(f"mae：{np.mean(losses_mae)}")
        print(f"mse：{np.mean(losses_mse)}")
        print(f"mape：{np.mean(losses_mape)}")

        return np.mean(losses_mae), np.mean(losses_mse), np.mean(losses_mape)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == '__main__':
    args = parse_args()
    log_path = './output.log'

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f) 

        print(args)
        mae, mse, mape = run_experiment(args)

        sys.stdout = original_stdout

