import os
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from dataloader import RecDataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
# from typing import List  


class ESCM2(nn.Module):  
    def __init__(self, sparse_feature_number, sparse_feature_dim, num_field,  
                 ctr_layer_sizes, cvr_layer_sizes, expert_num, expert_size,  
                 tower_size, counterfact_mode, feature_size):  
        super(ESCM2, self).__init__()  
        self.sparse_feature_number = sparse_feature_number  
        self.sparse_feature_dim = sparse_feature_dim  
        self.num_field = num_field  
        self.expert_num = expert_num  
        self.expert_size = expert_size  
        self.tower_size = tower_size  
        self.counterfact_mode = counterfact_mode  
        self.gate_num = 3 if counterfact_mode == "DR" else 2  
        self.feature_size = feature_size  
  
        self.embedding = nn.Embedding(  
            self.sparse_feature_number,  
            self.sparse_feature_dim,  
            padding_idx=0)  
        
        self.bn = nn.BatchNorm1d(num_features=self.feature_size)

        self.experts = nn.ModuleList([  
            nn.Linear(self.feature_size, self.expert_size, bias=True) for _ in range(self.expert_num)  
        ])  
          
        self.gates = nn.ModuleList([  
            nn.Linear(self.feature_size, self.expert_num, bias=True) for _ in range(self.gate_num)  
        ])  
          
        self.towers = nn.ModuleList([  
            nn.Linear(self.expert_size, self.tower_size, bias=True) for _ in range(self.gate_num)  
        ])  
          
        self.tower_outs = nn.ModuleList([  
            nn.Linear(self.tower_size,2 , bias=True) for _ in range(self.gate_num)  
        ])  
  
        # # 初始化权重
        # for m in self.modules():  
        #     if isinstance(m, nn.Linear):  
        #         nn.init.xavier_uniform_(m.weight)  
        #         nn.init.constant_(m.bias, 0.1)  
  
    def forward(self, inputs):  
        #inputs是一个列表，其中每个元素是一个形状为[batch_size, num_field]的张量  
        emb = []  
        for data in inputs:  
            feat_emb = self.embedding(data)  
            feat_emb = torch.sum(feat_emb, dim=1)  # sum pooling  
            emb.append(feat_emb)  
        concat_emb = torch.cat(emb, dim=1)  
        #print("concat_emb.shape:", concat_emb.shape) # shape=[N, 276]
        #concat_emb = inputs
        #concat_emb = self.bn(inputs) 
        expert_outputs = [F.relu(expert(concat_emb)) for expert in self.experts] 
        expert_concat = torch.reshape(torch.cat(expert_outputs, dim=1), [-1, self.expert_num, self.expert_size]) 

        output_layers = []  
        for i in range(self.gate_num):  
            gate_output = F.softmax(self.gates[i](concat_emb), dim=1)  
            gate_output = torch.reshape(gate_output, [-1, self.expert_num, 1])  
            weighted_expert = torch.sum(gate_output * expert_concat, dim=1)  
            tower_output = F.relu(self.towers[i](weighted_expert))  
            out = F.softmax(self.tower_outs[i](tower_output), dim=1)  
            out = torch.clamp(out, min=1e-15, max=1.0 - 1e-15)  
            output_layers.append(out)  
  
        ctr_out = output_layers[0]  
        cvr_out = output_layers[1]  
  
        ctr_prop_one = ctr_out[:, 1:2]  
        cvr_prop_one = cvr_out[:, 1:2]  
        ctcvr_prop_one = ctr_prop_one * cvr_prop_one  
        ctcvr_prop = torch.cat([1 - ctcvr_prop_one, ctcvr_prop_one], dim=1)  
  
        out_list = [ctr_out, ctr_prop_one, cvr_out, cvr_prop_one, ctcvr_prop, ctcvr_prop_one]  
        if self.counterfact_mode == "DR":  
            imp_out = output_layers[2]  
            imp_prop_one = imp_out[:, 1:2]  
            out_list.append(imp_prop_one)  
        
        # print("out_list:", out_list)
  
        return out_list  


@dataclass
class Config(object):
    '''
        config params
    '''
    train_data_dir: str = "/home/orange/orangeai/PaddleRec/datasets/ali-ccp/train_data"
    eval_data_dir: str = "/home/orange/orangeai/PaddleRec/datasets/ali-ccp/test_data"
    train_data_path: str = "/home/orange/orangeai/PaddleRec/datasets/ali-ccp/train_data/traindata_10w.csv"
    eval_data_path: str = "/home/orange/orangeai/PaddleRec/datasets/ali-ccp/test_data/testdata_10w.csv"
    model_save_checkpoint: str = "/home/orange/orangeai/ESCM2/output"
    batch_size: int = 4096
    learning_rate: float = 0.001
    lr_scheduler_epoch_size: int = 1
    epoches: int = 30
    interval_batch: int = 10
    device: str = "cuda"
    tensorboard_dir: str = "/home/orange/orangeai/ESCM2/output"
    num_workers: int = 0
    lr_milestones: list = field(default_factory=list) 
    lr_gamma: float = 0.1
    task_num: int = 3
    sep: int = ','
  


@dataclass
class ESCM2Config(Config):
    '''
        ESCM2 config class
    '''
    sparse_feature_number: int = 737946
    sparse_feature_dim: int = 12
    num_field: int = 23 # feature  num
    ctr_fc_sizes: list = field(default_factory=list)  #[256, 64]
    cvr_fc_sizes: list = field(default_factory=list)  #[256, 64]
    expert_num: int = 8
    expert_size: int = 16
    tower_size: int = 8
    counterfact_mode: str = "DR"
    feature_size: int = 276
    
    max_len: int = 3
    global_w: float = 0.5
    counterfactual_w: float = 0.5



class Pipeline(object):
    '''
        build a pipeline
    '''

    def build_dataloader(self, config: ESCM2Config):
        '''
            dataloader maybe a DataSet or IterableDataset
        '''
        if config is None:
                raise ValueError("config is None!!")
        
        batch_size = config.batch_size
        num_workers = config.num_workers

        train_file_list = [os.path.join(config.train_data_dir, x) for x in os.listdir(config.train_data_dir)]
        train_dataset = RecDataset(train_file_list, config)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )

        eval_file_list = [os.path.join(config.eval_data_dir, x) for x in os.listdir(config.eval_data_dir)]
        eval_dataset = RecDataset(eval_file_list, config)
        eval_dataloader = DataLoader(
            dataset=eval_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )

        return train_dataloader, eval_dataloader
    

    def build_opendataset_batch_inputs(self, batch_data, config:ESCM2Config):
        ''' 
            Task: build inputs of a batch 
            Args:
                batch_data: a batch data
                config: a hyper params config 
        '''
        max_len = config.max_len 
        sparse_tensor = [] # batch_data: [Tensor(shape=[1024, 3]) x 23, ... .... ,Tensor(shape=[1024, 1]) x 2]
        for data in batch_data[:-2]:
            sparse_tensor.append(torch.tensor(data, dtype=torch.long).view(-1, max_len)) # [1024, 3]

        ctr_label = torch.tensor(batch_data[-2], dtype=torch.long).view(-1, 1) # [1024, 1]
        ctcvr_label = torch.tensor(batch_data[-1], dtype=torch.long).view(-1, 1)
        # print("sparse_tensor:", sparse_tensor)
        return sparse_tensor, ctr_label, ctcvr_label


    def build_model(self, config: ESCM2Config):
        '''
            build a ESCM2 model
        '''
        self.model = ESCM2(
            sparse_feature_number=config.sparse_feature_number,
            sparse_feature_dim=config.sparse_feature_dim,
            num_field=config.num_field,
            ctr_layer_sizes=config.ctr_fc_sizes,
            cvr_layer_sizes=config.cvr_fc_sizes,
            expert_num=config.expert_num,
            expert_size=config.expert_size,
            tower_size=config.tower_size,
            counterfact_mode=config.counterfact_mode,
            feature_size=config.feature_size
        )
    
        return self.model
    

    def build_loss(self, config: ESCM2Config, ctr_out_one, ctr_clk, ctcvr_prop_one, ctcvr_buy, cvr_out_one, out_list):  
            '''
            ESCM2 loss function design including the Doubly Roubst Loss 
            '''

            # Cast labels to float  
            ctr_clk_float = ctr_clk.float()  
            ctcvr_buy_float = ctcvr_buy.float()  
            # ctr_out_one.shape: torch.Size([1024, 1]) ctr_clk_float.shape torch.Size([1024])
            print("ctr_out_one.shape:", ctr_out_one.shape, "ctr_clk_float.shape", ctr_clk_float.shape) 
            # Compute losses  
            loss_ctr = F.binary_cross_entropy_with_logits(ctr_out_one, ctr_clk_float)  
            loss_cvr = F.binary_cross_entropy_with_logits(cvr_out_one, ctcvr_buy_float)  
            loss_ctcvr = F.binary_cross_entropy_with_logits(ctcvr_prop_one, ctcvr_buy_float)  
    
            if config.counterfact_mode == "DR":  
                loss_cvr = self.counterfact_dr(loss_cvr, ctr_clk_float, ctr_out_one, out_list[6])  
            else:  
                loss_cvr = self.counterfact_ipw(loss_cvr, ctr_clk, ctr_clk_float, ctr_out_one)  
    
            # Compute the final cost, weighted by counterfactual and global weights  
            loss = loss_ctr + loss_cvr * config.counterfactual_w + loss_ctcvr * config.global_w   # cal batch mean loss?
            loss_list = [loss_ctr.mean(), loss_cvr.mean(), loss_ctcvr.mean()]

            return loss.mean(), loss_list
    

    def counterfact_ipw(self, loss_cvr, ctr_clk, ctr_clk_float, ctr_out_one):  
        '''
            ipw loss
        '''
        ctr_num = ctr_clk.sum()  
        PS = ctr_out_one * ctr_num.float()  
        PS = torch.clamp(PS, min=1e-6)  # Avoid division by zero  
        IPS = 1.0 / PS  
        IPS = torch.clamp(IPS, min=-15, max=15)  
        #IPS.requires_grad_(False)  # Stop gradient  
        IPS = IPS.detach()
        batch_size = float(ctr_clk.size(0))  
        IPS_scaled = IPS * batch_size  
        loss_cvr_weighted = loss_cvr * IPS_scaled  
        loss_cvr_final = (loss_cvr_weighted * ctr_clk_float).mean()  
        return loss_cvr_final  
  

    def counterfact_dr(self, loss_cvr, ctr_clk_float, ctr_out_one, imp_out):  
        '''
            Doubly Roubst loss 
        '''
        e = loss_cvr - imp_out  
        ctr_out_one_clamped = torch.clamp(ctr_out_one, min=1e-6) 
        IPS = ctr_clk_float / ctr_out_one_clamped  
        IPS = torch.clamp(IPS, min=-15, max=15)  
        # IPS.requires_grad_(False)  
        IPS =  IPS.detach()
        loss_error_second = e * IPS  
        loss_error = imp_out + loss_error_second  
        loss_imp = e**2  
        loss_imp_weighted = loss_imp * IPS  
        loss_dr = loss_error + loss_imp_weighted  
  
        return loss_dr.mean()  
    

    def build_optimizer(self, model: ESCM2, config:ESCM2Config):  
        lr = config.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma)

        return optimizer, scheduler
  

    def build_mutiltask(self):  
        # from sklearn.metrics import roc_auc_score
        # auc_ctr_metric = roc_auc_score
        # auc_cvr_metric = roc_auc_score
        # auc_ctcvr_metric = roc_auc_score
        task_list_name = ["ctr_task", "cvr_task", "ctcvr_task"]  
        # metrics_list = [auc_ctr_metric, auc_cvr_metric, auc_ctcvr_metric]  
        return  task_list_name  


    def train_forward(self, model, batch_data, config, metrics_list=None):  
        '''
            Args: 
                sparse_tensor.shape:  [Tensor(shape=[1024, 3]) x 23]
                label_ctr.shape: Tensor(shape=[1024, 1])
                label_ctcvr.shape:  Tensor(shape=[1024, 1])
                metrics_list: [[], [], []]
        '''
        sparse_tensor, label_ctr, label_ctcvr = self.build_opendataset_batch_inputs(batch_data, config)
        sparse_tensor, label_ctr, label_ctcvr  = [s.to(config.device) for s in sparse_tensor], label_ctr.to(config.device), label_ctcvr.to(config.device)

  
        # model.train() 
        out_list = model(sparse_tensor) 
        #print("out_list:", len(out_list))
        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one, imp_score = out_list  
        loss, loss_list = self.build_loss(config, ctr_out_one, label_ctr, ctcvr_prop_one, label_ctcvr, cvr_out_one, out_list)  
        #print("loss:", loss) # loss: tensor(1.6060, device='cuda:0', grad_fn=<MeanBackward0>)
        #print("loss_list:", loss_list) # loss_list: [tensor(0.9356, device='cuda:0', grad_fn=<MeanBackward0>), tensor(0.5790, device='cuda:0', grad_fn=<MeanBackward0>), tensor(0.7618, device='cuda:0', grad_fn=<MeanBackward0>)]
        # record metrics y_pred and y_true
        # print("ctr_out:", ctr_out)
        # print("ctr_out_one:", ctr_out_one)
        # print("torch.sigmoid(ctr_out)", torch.sigmoid(ctr_out))
        if metrics_list: 
            metrics_list[0].append(torch.cat((ctr_out_one, label_ctr), dim=1)) # [(N, 2), (N, 2),...]
            metrics_list[1].append(torch.cat((cvr_out_one, label_ctcvr), dim=1)) # [(N, 2), (N, 2),...]
            metrics_list[2].append(torch.cat((ctcvr_prop_one, label_ctcvr), dim=1)) # [(N, 2), (N, 2),...]
        
        return loss, loss_list, metrics_list


    def infer_forward(self, model, batch_data, config, metrics_list):  
        '''
            Args:
                metrics_list: [[], [], []]
        '''
        # model.eval()   
        
        sparse_tensor, label_ctr, label_ctcvr = self.build_opendataset_batch_inputs(batch_data, config)
        sparse_tensor, label_ctr, label_ctcvr  = sparse_tensor.to(config.device), label_ctr.to(config.device), label_ctcvr.to(config.device)
  
        out_list = model(sparse_tensor) 
        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one, _ = out_list  
        loss, loss_list = self.build_loss(config, ctr_out_one, label_ctr, ctcvr_prop_one, label_ctcvr, cvr_out_one, out_list)  
        # record metrics y_pred and y_true

        metrics_list[0].append(torch.cat((ctr_out_one, label_ctr), dim=1)) # [(N, 2), (N, 2),...]
        metrics_list[1].append(torch.cat((cvr_out_one, label_ctcvr), dim=1)) # [(N, 2), (N, 2),...]
        metrics_list[2].append(torch.cat((ctcvr_prop_one, label_ctcvr), dim=1)) # [(N, 2), (N, 2),...]

        return loss, loss_list, metrics_list
    
    