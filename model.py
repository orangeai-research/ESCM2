#   Copyright (c) 2024 orangerec Authors. All Rights Reserved.

import torch  
import torch.nn as nn  
import torch.nn.functional as F  

from torch.utils.data.dataloader import  DataLoader
from ESCM2.dataloader import RecDataset
  
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
            nn.Linear(self.tower_size, 2, bias=True) for _ in range(self.gate_num)  
        ])  
  
        # 初始化权重（如果需要的话，可以在这里自定义初始化方式）  
        for m in self.modules():  
            if isinstance(m, nn.Linear):  
                nn.init.xavier_uniform_(m.weight)  
                nn.init.constant_(m.bias, 0.1)  
  
    def forward(self, inputs):  
        # 假设inputs是一个列表，其中每个元素是一个形状为[batch_size, num_field]的张量  
        emb = []  
        for data in inputs:  
            feat_emb = self.embedding(data)  
            feat_emb = torch.sum(feat_emb, dim=1)  # sum pooling  
            emb.append(feat_emb)  
        concat_emb = torch.cat(emb, dim=1)  
  
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
  
        return out_list  


@dataclass
class Config(object):
    '''
        config params
    '''
    train_data_dir: str = ""
    eval_data_dir: str = ""
    model_save_checkpoint: str = ""
    batch_size: int = 1024
    learning_rate: float = 0.001
    lr_scheduler_epoch_size: int = 1
    epoches: int = 3
    interval_batch: int = 10
    device: str = "cuda"
    tensorboard_dir: str = ""
    num_workers:int = 0

@dataclass
class ESCM2Config(Config):
    '''
        ESCM2 config class
    '''
    def __init__(self):
        super(ESCM2Config, self).__init__()
        sparse_feature_number = 737946
        sparse_feature_dim = 12
        num_field = 23 # feature  num
        ctr_fc_sizes = [256, 64]
        cvr_fc_sizes = [256, 64]
        expert_num = 8
        expert_size = 16
        tower_size = 8
        counterfact_mode = "DR"
        feature_size = 276
        
        max_len = 3
        global_w = 0.5
        counterfactual_w = 0.5
        learning_rate =  0.001




class Pipeline(object):
    '''
        build a pipeline
    '''

    def build_dataloader(self, config: ESCM2Config):
        '''
            dataloader maybe a DataSet or IterableDataset
        '''
        batch_size = config.batch_size
        num_workers = num_workers
        train_dataset = RecDataset(file_list=config.train_data_dir, config=config)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )

        eval_dataset = RecDataset(file_list=config.eval_data_dir, config=config)
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
            sparse_tensor.append(torch.tensor(data, dtype==torch.long)).view(-1, max_len) # [1024, 3]

        ctr_label = torch.tensor(batch_data[-2], dtype=torch.long).view(-1, 1) # [1024, 1]
        ctcvr_label = torch.tensor(batch_data[-1], dtype=torch.long).view(-1, 1)
        return sparse_tensor, ctr_label, ctcvr_label


    def build_model(self, config: ESCM2Config):
        '''
        
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
    
        return model 
    

    def build_loss(self, config: ESCM2Config, ctr_out_one, ctr_clk, ctcvr_prop_one, ctcvr_buy, cvr_out_one, out_list):  
            '''
            ESCM2 loss function design including the Doubly Roubst Loss 
            '''

            # Cast labels to float  
            ctr_clk_float = ctr_clk.float()  
            ctcvr_buy_float = ctcvr_buy.float()  
    
            # Compute losses  
            loss_ctr = F.binary_cross_entropy_with_logits(ctr_out_one, ctr_clk_float)  
            loss_cvr = F.binary_cross_entropy_with_logits(cvr_out_one, ctcvr_buy_float)  
            loss_ctcvr = F.binary_cross_entropy_with_logits(ctcvr_prop_one, ctcvr_buy_float)  
    
            if config.counterfact_mode == "DR":  
                loss_cvr = self.counterfact_dr(loss_cvr, ctr_clk_float, ctr_out_one, out_list[6])  
            else:  
                loss_cvr = self.counterfact_ipw(loss_cvr, ctr_clk, ctr_clk_float, ctr_out_one)  
    
            # Compute the final cost, weighted by counterfactual and global weights  
            cost = loss_ctr + loss_cvr * self.counterfactual_w + loss_ctcvr * self.global_w   # cal batch mean loss?
            cost = cost.mean()
            return cost 
    

    def counterfact_ipw(self, loss_cvr, ctr_clk, ctr_clk_float, ctr_out_one):  
        '''
            ipw loss
        '''
        ctr_num = ctr_clk.sum()  
        PS = ctr_out_one * ctr_num.float()  
        PS = torch.clamp(PS, min=1e-6)  # Avoid division by zero  
        IPS = 1.0 / PS  
        IPS = torch.clamp(IPS, min=-15, max=15)  
        IPS.requires_grad_(False)  # Stop gradient  
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
        IPS.requires_grad_(False)  
  
        loss_error_second = e * IPS  
        loss_error = imp_out + loss_error_second  
  
        loss_imp = e**2  
        loss_imp_weighted = loss_imp * IPS  
        loss_dr = loss_error + loss_imp_weighted  
  
        return loss_dr.mean()  
    

    def create_optimizer(self, model: ESCM2, config:ESCM2Config):  
        lr = config.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
        return optimizer  
  

    def create_metrics(self):  
        metrics_list_name = ["auc_ctr", "auc_cvr", "auc_ctcvr"]  
        auc_ctr_metric = None  
        auc_cvr_metric = None  
        auc_ctcvr_metric = None  
        metrics_list = [auc_ctr_metric, auc_cvr_metric, auc_ctcvr_metric]  
        return metrics_list, metrics_list_name  


    def train_forward(self, model, metrics_list, batch_data, config):  
        sparse_tensor, label_ctr, label_ctcvr = self.create_feeds(batch_data)  
  
        model.train() 
        out_list = model(sparse_tensor) 
        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = out_list  
        loss = self.build_loss(config, ctr_out_one, label_ctr, ctcvr_prop_one, label_ctcvr, cvr_out_one, out_list)  

  
        print_dict = {'loss': loss.item()}  
        return loss, metrics_list, print_dict  
  
    def infer_forward(self, model, metrics_list, batch_data, config):  
        model.eval()   
        with torch.no_grad(): 
            sparse_tensor, label_ctr, label_ctcvr = self.create_feeds(batch_data)  
            out_list = model(sparse_tensor)  

            ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = out_list  
  
  
        return metrics_list, None