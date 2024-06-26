'''
   Train and Eval Pipeline
'''

import os 
from datetime import datetime
import torch
import logging 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  
from model import  ESCM2Config, Pipeline


def main():
    # logging  
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
    logger = logging.getLogger(__name__)  

    # config 
    config = ESCM2Config(
        lr_milestones = [2, 5],
        ctr_fc_sizes = [512, 64],
        cvr_fc_sizes = [512, 64],
    )

    # pipeline
    pipeline = Pipeline()

    #device 
    if torch.cuda.is_available():  
        print("CUDA is available!")  
        device = torch.device("cuda")  
    else:  
        print("No money offer a CUDA")  
        device = torch.device("cpu")
    config.device = device 

    # TensorBoard launch
    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    # build train_dataloader, eval_dataloader 
    logger.info("load data......")
    train_dataloader, eval_dataloader = pipeline.build_dataloader(config)

    # build model 
    model = pipeline.build_model(config)
    optimizer, scheduler = pipeline.build_optimizer(model, config)
    task_name_list = pipeline.build_mutiltask()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 添加时间戳生成
    epoches = config.epoches
    EVAL_BEST_AUC = 0.0
    model = model.to(device)
    for epoch in range(epoches):
        
        train_iter_batch_num = 0 # record batch num in one epoch
        train_mini_batch_loss = 0.0 # mini batch loss
        train_mini_batch_loss_list = [0.0 for i in range(config.task_num)] # multi-task mini batch loss [0.0, 0.0, 0.0]
        train_loss = 0.0 # one whole epoch loss
        train_mini_batch_metrics_list = tuple(list() for i in range(config.task_num)) # tuple expression save memory efficientlly

        metrics_list = tuple(list() for i in range(config.task_num)) # ([], [], []) or [[]]
        eval_iter_batch_num = 0 # record batch num in one epoch
        eval_mini_batch_loss = 0.0 # mini batch loss
        eval_mini_batch_loss_list = [0.0 for i in range(config.task_num)] # multi-task mini batch loss
        eval_loss = 0.0 # one whole epoch loss
        
        model.train()
        for i, batch_data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1} / {epoches}")):
            optimizer.zero_grad()
            loss, loss_list, train_mini_batch_metrics_list  = pipeline.train_forward(model, batch_data, config, train_mini_batch_metrics_list) # train_mini_batch_metrics_list: ([(N, 2),...], [(N, 2),...],, [(N, 2),...])
            loss.backward()
            optimizer.step()
            train_mini_batch_loss += loss.item() # record mini batch loss
            train_mini_batch_loss_list  = [a + b.item() for a, b in zip(train_mini_batch_loss_list, loss_list)] # record multi-task mini batch loss [[f1], [f2], [f3]]
            train_loss += loss.item() # one whole epoch train loss
            train_iter_batch_num += 1
            if i % config.interval_batch == 0:
                # logging metrics
                res = tuple(torch.cat(metric, dim=0) for metric in train_mini_batch_metrics_list) # ([B,2], [B,2], [B,2])
                # print("res:", res)
                # print("res:", res[0][:, 1], res[0][:, 0])
                '''
                metrics_dict:
                    {
                        "ctr_task" : [float(auc), float(accuracy), float(precision), float(recall), float(f1)]
                        "cvr_task" : [float(auc), float(accuracy), float(precision), float(recall), float(f1)]
                        "ctcvr_task" : [float(auc), float(accuracy), float(precision), float(recall), float(f1)]
                    }
                '''
                train_metrics_dict = {task_name : caculate_metrics(pred_label[:, 1].cpu().detach().numpy(), pred_label[:, 0].cpu().detach().numpy()) 
                                      for task_name, pred_label in zip(task_name_list, res)}
                str_metrics = ''.join(['Epoch{}: Task: {} AUC: {:.4f} ACC: {:.4f} Precision: {:.4f} Recall: {:.4f} F1-Score: {:.4f} {}'.format(epoch + 1, task_name, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], "\n" ) 
                                       for task_name, metrics in train_metrics_dict.items()])
                # logging loss
                str_loss = ' '.join(['Loss{}: {:.4f}'.format(i, train_mini_batch_loss_list[i] /config.interval_batch ) for i in range(len(train_mini_batch_loss_list))])
                train_batch_avg_loss = train_mini_batch_loss / config.interval_batch
                logger.info('Epoch: [{}/{}], Step: [{}], Lr: {:.6f}, Total Loss: {:.6f}, MTL Loss: {}, {} Metrics: {}'.format(epoch + 1, epoches, i + 1, optimizer.param_groups[0]['lr'], 
                                                                                              train_batch_avg_loss, str_loss, "\n", str_metrics))
                # record tensorboard scalar
                tb_x = epoch * train_iter_batch_num + i + 1
                writer.add_scalar('Loss/Train', train_batch_avg_loss, tb_x)
                # writer.add_scalar('AUC/Train', auc, tb_x)  
                # writer.add_scalar('Accuracy/Train', accuracy, tb_x)  
                # writer.add_scalar('Precision/Train', precision, tb_x)  
                # writer.add_scalar('Recall/Train', recall, tb_x)  
                # writer.add_scalar('F1 Score/Train', f1, tb_x)  
                train_mini_batch_loss = 0.0 # reset mini batch loss
                train_mini_batch_loss_list = [0.0 for i in range(config.task_num)] # reset mini batch multi-task loss list 
                train_mini_batch_metrics_list = tuple(list() for i in range(config.task_num)) 

        scheduler.step()
        # logger.info(f'Epoch [{epoch + 1}/{epoches}] Train Loss: {train_loss / train_iter_batch_num}  Train learning rate: {optimizer.param_groups[0]['lr']}')  
        logger.info('Epoch: [{}/{}] Test:'.format(epoch+1, epoches))
        model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(eval_dataloader, desc=f"Epoch {epoch + 1} / {epoches}")):
                loss, loss_list, metrics_list = pipeline.infer_forward(model, batch_data, config, metrics_list) # metrics_list: [[(N, 2),...], [(N, 2),...],, [(N, 2),...]]
                eval_mini_batch_loss += loss.item() 
                eval_mini_batch_loss_list = [a + b.item() for a, b in zip(eval_mini_batch_loss_list, loss_list)]
                eval_loss += loss.item()
                eval_iter_batch_num += 1
            
            res = tuple(torch.cat(metric, dim=0) for metric in metrics_list) # [[B, 2], [B, 2], [B, 2]]
            metrics_dict  = {task_name : caculate_metrics(pred_label[:, 1].cpu().detach().numpy(), pred_label[:, 0].cpu().detach().numpy()) 
                             for task_name, pred_label in zip(task_name_list, res)}
            
            str_loss = ' '.join(['Loss{}: {:.4f}'.format(i, eval_mini_batch_loss_list[i] / eval_iter_batch_num) for i in range(len(eval_mini_batch_loss_list))])
            str_metrics = ''.join(['Epoch{}: Task:{} AUC{:.4f} ACC{:.4f} Precision{:.4f} Recall{:.4f} F1-Score{:.4f}'.format(epoch, task_name, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4] ) 
                                   for task_name, metrics in metrics_dict.items()])
            logger.info('Epoch: [{}/{}],  MTK Loss: {}, Metrics: {}'.format(epoch + 1, epoches, str_loss, str_metrics))

            # record tensorboard  
            for task_name, metrics in metrics_dict.items():
                            writer.add_scalar('{}_Loss/eval'.format(task_name), eval_loss / eval_iter_batch_num, epoch)  
                            writer.add_scalar('{}_AUC/eval'.format(task_name), metrics[0], epoch)  
                            writer.add_scalar('{}_Accuracy/eval'.format(task_name), metrics[1], epoch)  
                            writer.add_scalar('{}_Precision/eval'.format(task_name), metrics[1], epoch)  
                            writer.add_scalar('{}_Recall/eval'.format(task_name), metrics[1], epoch)  
                            writer.add_scalar('{}_F1-Score/eval'.format(task_name), metrics[1], epoch)  
            
            # model save
            aim_task_auc = list(metrics_dict.values())[-1][0] # find the ctcvr-task auc
            if aim_task_auc > EVAL_BEST_AUC:
                EVAL_BEST_AUC = aim_task_auc
                if config.model_save_checkpoint:
                    model_path = os.path.join(config.model_save_checkpoint, 'model_{}_{}.pt'.format(timestamp, epoch))
                    jit_model_path = os.path.join(config.model_save_checkpoint, 'jit_model_{}_{}.pt'.format(timestamp, epoch))
                    torch.save(model.state_dict(), model_path)
                    logger.info('Model performance improved, model saving checkpoint successfully!') 
                    try:
                        torch.jit.save(torch.jit.script(model), jit_model_path) 
                    except ValueError as e :
                         print("mdoel jit save fail:", e)
                    logger.info('Model performance improved. jit model saving checkpoint successfully!') 
                else:
                    raise ValueError("model checkpoint path is null, please check your checkpoint path!!")
            
    writer.close()


def caculate_metrics(y_true:list, y_pred:list)->list:
    """
        caculate AUC, ACC, Precision, Recall, F1-Score, etc...
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score  
    y_true = y_true.astype(int)
    auc = roc_auc_score(y_true, y_pred) 
    print("sum of y_true", y_true.sum(), "y_true.shape:", y_true.shape)
    # print("y_true:", y_true, "y_pred:", y_pred, "auc:", auc)
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))  
    precision = precision_score(y_true, (y_pred > 0.5).astype(int))  
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))  
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))  
    metrics_list = [float(auc), float(accuracy), float(precision), float(recall), float(f1)]
    return metrics_list

if __name__ == "__main__":
    main()

