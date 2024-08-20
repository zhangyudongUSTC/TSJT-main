import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from tqdm import tqdm
from train import main_loop
from Models.meta_gwn import gwnet


class NPM_model(nn.Module):
    def __init__(self, 
                 num_node,
                 in_len,
                 in_dim,
                 STModel,
                 out_dim=128, 
                 size=150):
        super().__init__()
        self.npm = NPM(num_node,
                       in_len,
                       in_dim,
                       out_dim=out_dim, 
                       size=size)
        self.stmodel = STModel

        def forward(self, x, a):
            z, ak = self.npm(x, a)
            return self.stmodel(z, ak)

parser = argparse.ArgumentParser(description='TSJT-based')
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='metr-la', type=str)
parser.add_argument('--source_epochs', default=200, type=int)
parser.add_argument('--source_lr', default=1e-3, type=float)
parser.add_argument('--target_epochs', default=120, type=int)
parser.add_argument('--target_days', default=3, type=int)
parser.add_argument('--target_lr', default=1e-3, type=float)
parser.add_argument('--model', default='GRU', type=str)

# parser.add_argument('--memo', default='revise', type=str)
args = parser.parse_args()


if __name__ == '__main__':

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.load(f, yaml.FullLoader)

    torch.manual_seed(7)

    data_args, task_args, model_args = config['data'], config['task'], config['model']
    num_node = 627
    in_dim = config['model']['node_feature_dim']
    out_dim = 128
    in_len = config['task']['his_num']
    out_len = config['task']['pred_num']
    print(type(in_dim), in_len, out_len)
    print(task_args.keys())
    source_dataset = traffic_dataset(data_args, task_args, "source", test_data=args.test_dataset)
    source_dataloader = DataLoader(source_dataset, batch_size=task_args['source_batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    target_dataset = traffic_dataset(data_args, task_args, "target", test_data=args.test_dataset, target_days=args.target_days)
    target_dataloader = DataLoader(target_dataset, batch_size=task_args['target_batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    test_dataset = traffic_dataset(data_args, task_args, "test", test_data=args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=task_args['target_batch_size'], shuffle=True, num_workers=8, pin_memory=True)

    # define stmodel --------
    # removed
    stmodel = None
    #------------------------

    model = NPM_model(num_node=num_node, 
                      in_len=in_len,
                      in_dim=in_dim,
                      STModel=stmodel
                      out_dim=out_dim,
                      size=150).to(device=args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.source_lr)
    loss_criterion = nn.MSELoss()


    main_loop(model, source_dataset, traffic_dataset, optimizer, None, loss_criterion, config['task'].epoch, args.device)