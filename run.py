import yaml
import json
import argparse
import torch
import time
import copy
import wandb
import os

from utils.get_dataloader import get_dataloader
from s2kt import S2KT
from dkt import DKT
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
parser = argparse.ArgumentParser(description='args')
parser.add_argument('--config_path', type=str, default='config.yaml')
parser.add_argument('--log_path', type=str, default='assist09.log')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(dataloader, seq_model, optimizer, device, args, epoch):
    seq_model.train()
    loss_list = []
    # for batch in tqdm(dataloader, desc='training ...'):
    for batch in dataloader:
        question_seq, answer_seq, seq_len = batch
        question_seq = question_seq.to(device)
        answer_seq = answer_seq.to(device)
        seq_len = seq_len.to(device)

        loss = seq_model.calculate_loss(question_seq, answer_seq, seq_len)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = sum(loss_list)/len(loss_list)
    print('epoch: {}, loss: {}'.format(epoch, train_loss))
    with open(args.log_path, 'a') as f:
        f.write('epoch: {}, train loss: {}, '.format(epoch, train_loss))

def eval(dataloader, seq_model, device, args):
    seq_model.eval()
    loss_list = []
    with torch.no_grad(): 
        for batch in dataloader:
            question_seq, answer_seq, seq_len = batch
            question_seq = question_seq.to(device)
            answer_seq = answer_seq.to(device)
            seq_len = seq_len.to(device)
            loss = seq_model.calculate_loss(question_seq, answer_seq, seq_len)
            loss_list.append(loss.item())
    eval_loss = sum(loss_list)/len(loss_list)
    print('evaluate loss: {}'.format(eval_loss))
    with open(args.log_path, 'a') as f:
        f.write('evaluate loss: {}\n'.format(eval_loss))
    return eval_loss


sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': 
    {
        'lmd1': {'values':[1.0, 0.7, 0.5, 0.3, 0.0]},
        'lmd2': {'values':[1.0, 0.7, 0.5, 0.3, 0.0]},
        'lmd3': {'values':[1.0, 0.5, 0]},
        'lmd4': {'values':[1.0, 0.5, 0]},
        # 'lmd1': {'values':[1.0]},
        # 'lmd2': {'values':[1.0, 0.0]},
        # 'lmd3': {'values':[1.0]},
        # 'lmd4': {'values':[0.5]},
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='S2KT')
def main():
    run = wandb.init('S2KT')
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_folder = os.path.join('./data', config['dataset'])
    # print(config)
    # print(data_folder)
    problem_id_hashmap = json.load(open(f'{data_folder}/problem_id_hashmap.json', "r", encoding="utf-8"))
    config['n_questions'] = len(problem_id_hashmap)
    # config['lmd1'] = 1.0
    # config['lmd2'] = 0.5
    # config['lmd3'] = 0.5
    # config['lmd4'] = 1.0
    wandb.config.update(config)
    config = wandb.config
    
    # load data
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(dataset=config['dataset'], max_step=config['max_seq_length'], batch_size = config['batch_size'], mode='problem')
    
    # pretrain
    s2kt_model = S2KT(config, device=device)
    optimizer = torch.optim.Adam(s2kt_model.parameters(), lr=config['learning_rate'])
    
    best_s2kt_model = s2kt_model
    best_loss = 9999
    stale, patience= 0, 7
    for epoch in range(1, config['epochs'] + 1):
        train(train_dataloader, s2kt_model, optimizer, device, args, epoch)
        eval_loss = eval(dev_dataloader, s2kt_model, device, args)
       
        if eval_loss < best_loss:
            stale = 0
            best_loss = eval_loss
            best_s2kt_model = copy.deepcopy(s2kt_model)
        else:
            stale += 1
            if stale > patience:
                print(
                    f"No improvment {patience} consecutive epochs, early stopping"
                )
                break

    s2kt_model = best_s2kt_model

    # downtask 
    dkt_model = DKT(input_size=config['hidden_size'] * 2,
          num_questions=len(problem_id_hashmap),
          hidden_size=config['hidden_size'],
          num_layers=1,
          embedding=s2kt_model.general_question_embedding + s2kt_model.question_embedding.weight.data,
          max_steps=config['max_seq_length'],
          log_path=args.log_path,
          seq_model = s2kt_model
          )

    # dkt train
    dkt_model.train(train_data_loader=train_dataloader,
            test_data=dev_dataloader,
            epoch=200)
    
    # dkt test 
    dkt_model.dkt_model = dkt_model.best_dkt
    auc, acc = dkt_model.eval(test_dataloader)
    wandb.log({
        'auc': auc,
        'acc': acc
    })
    print('On test sest...')
    print("auc: %.6f" % auc)
    print("acc: %.6f" % acc)
    with open(args.log_path, 'a', encoding='utf-8') as f:
        f.write('On test sest...\n')
        f.write("auc: %.6f\n" % auc)
        f.write("acc: %.6f\n" % acc)
    
    datatime = time.strftime('%Y-%m-%d-%H-%M-%S')
    # save s2kt model, dkt model and corresponding config
    s2kt_model_path = os.path.join('models', config['dataset'] + f'_s2kt_{datatime}.params')
    dkt_model_path = os.path.join('models', config['dataset'] + f'_dkt_{datatime}.params')
    config_path = os.path.join('models', config['dataset'] + f'_config_{datatime}.yaml')
    torch.save(s2kt_model.state_dict(), s2kt_model_path)
    dkt_model.save(dkt_model_path)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    with open(args.log_path, 'a') as f:
        f.write(f's2kt_model path: {s2kt_model_path}, dkt_model path: {dkt_model_path}, config path: {config_path}\n')
# main()
# Start sweep job.
wandb.agent(sweep_id, function=main)