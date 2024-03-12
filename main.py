import os
import pandas as pd
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from dgllife.utils import EarlyStopping

from dataset import  collate,dataprocess
from utils import set_random_seed, evaluate
from model import Model
import config
import warnings
import pickle
warnings.filterwarnings("ignore")


def train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper):
    batch2=[]
    for batch1 in train_loader:
        batch2.append(batch1.to(args.device))
    batch3 = []

    for batch4 in val_loader:
        batch3.append(batch4.to(args.device))

    for epoch in range(args.epoch):
        model.train()
        # one_batch_bar = tqdm(train_loader, ncols=100)
        # one_batch_bar.set_description(f'[iter:{args.iter},epoch:{epoch + 1}/{args.epoch}]')
        cur_lr = optimizer.param_groups[0]["lr"]
        res1=[]
        for i, batch1 in enumerate(batch2):

            labels = batch1.labels


            pred,loss1 = model(batch1.padded_smiles_batch, batch1, batch1.fps_t)
            AC, F1, SN, SP, CCR, MCC = evaluate(labels.unsqueeze(1), pred)
            res1.append([AC, F1, SN, SP, CCR, MCC])

            # acc, precision, recall, f1score, acc_weight = evaluate(labels, pred)
            loss = loss_func(pred, labels.unsqueeze(1))+loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # one_batch_bar.set_postfix(dict(
            #     loss=f'{loss.item():.5f}',
            #     acc=f'{acc * 100:.2f}%')
        train_results = pd.DataFrame(res1, columns=['AC', 'F1', 'SN', 'SP', 'CCR', 'MCC'])
        r1 = train_results.mean()


        print(
            f"epoch:{epoch}---train---AC:{r1['AC']}---F1:{r1['F1']}---SN:{r1['SN']}---SP:{r1['SP']}---CCR:{r1['CCR']}---MCC:{r1['MCC']}-"
                 )

        scheduler.step()
        model.eval()
        res = []

        with torch.no_grad():
            for batch in batch3:
                # batch = batch.to(args.device)
                labels = batch.labels
                padded_smiles_batch = batch.padded_smiles_batch

                fps_t = batch.fps_t
                pred,loss1 = model(padded_smiles_batch, batch, fps_t)
                AC, F1, SN, SP, CCR,MCC = evaluate(labels.unsqueeze(1), pred)
                res.append([AC, F1, SN, SP,CCR,MCC])
        val_results = pd.DataFrame(res, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = val_results.mean()
        print(
            f"epoch:{epoch}---validation---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}-"
            f"---lr:{cur_lr}")
        early_stop = stopper.step(r['AC'], model)
        if early_stop:
            break


def main(args):
    source = 'Cav1.2'  # Cav1.2/hERG/Nav1.5
    datas = 'cav'  # cav/herg/nav
    dataspathmulu='./data/' + source + '/'
    data_path = './data/' + source + '/data_' + datas + '_dev.csv'
    data_path_test = './data/' + source + '/eval_set_' + datas + '_60.csv'
    data_path_test1 = './data/' + source + '/eval_set_' + datas + '_70.csv'
    data_index = pd.read_csv('./data/' + source + '/data_' + datas + '_dev.csv')
    # data_index_test=pd.read_csv('./data/Cav1.2/eval_set_cav_60.csv')
    if os.path.exists(dataspathmulu+'dataset.pkl'):
        with open(dataspathmulu+'dataset.pkl', 'rb') as file:
            dataset = pickle.load(file)
        with open(dataspathmulu+'test_dataset.pkl', 'rb') as file:
            test_dataset = pickle.load(file)
        with open(dataspathmulu+'test_dataset1.pkl', 'rb') as file:
            test_dataset1 = pickle.load(file)
    else:
        dataset = dataprocess(data_path,dataspathmulu+'dataset.pkl')
        test_dataset = dataprocess(data_path_test,dataspathmulu+'test_dataset.pkl')
        test_dataset1 = dataprocess(data_path_test1,dataspathmulu+'test_dataset1.pkl')


    data_index_train=data_index.index[data_index['USED_AS'] == 'Train'].tolist()
    # true_numbers = [num for num, flag in data_index_train if flag]
    data_index_Validation=data_index.index[data_index['USED_AS'] == 'Validation'].tolist()

    # data_index = []
    # file_name = "data_index.txt"
    # with open('./data' + "/" + file_name, "r") as f:
    #     for line in f.readlines():
    #         line = eval(line)
    #         data_index.append(line)

    train_dataset = Subset(dataset, data_index_train)
    validate_dataset = Subset(dataset,data_index_Validation)
    # test_dataset = Subset(dataset_test,)
    n_feats = 84

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate)
    val_loader = DataLoader(validate_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader60 = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader70 = DataLoader(test_dataset1, batch_size=args.batch_size, collate_fn=collate)
    mean_results = []
    mean_results1 = []
    for iteration in range(args.iterations):
        args.iter = iteration
        model = Model( in_feats=n_feats, hidden_feats=args.hidden_feats,
                    rnn_embed_dim=args.rnn_embed_dim, blstm_dim=args.rnn_hidden_dim, blstm_layers=args.rnn_layers,
                    fp_2_dim=args.fp_dim, dropout=args.p, num_heads=args.head, device=args.device).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        stopper = EarlyStopping(mode='higher', filename=f'{args.output}/net_{iteration}.pkl', patience= 50)
        loss_func = torch.nn.BCEWithLogitsLoss()
        train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper)
        stopper.load_checkpoint(model)
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        total_preds1 = torch.Tensor()
        total_labels1 = torch.Tensor()
        res60 = []
        res70 = []
        with torch.no_grad():
            for batch in test_loader60:
                batch = batch.to(args.device)
                labels = batch.labels
                padded_smiles_batch = batch.padded_smiles_batch
                fps_t = batch.fps_t
                pred,loss1= model(padded_smiles_batch, batch, fps_t)
                total_preds = torch.cat((total_preds, pred.cpu()), 0)
                total_labels = torch.cat((total_labels, labels.cpu()), 0)

            AC, F1, SN, SP, CCR,MCC  = evaluate(total_labels.unsqueeze(1), total_preds)
            res60.append([AC, F1, SN, SP, CCR,MCC])
        # with torch.no_grad():

            for batch in test_loader70:
                batch = batch.to(args.device)
                labels = batch.labels
                padded_smiles_batch = batch.padded_smiles_batch

                fps_t = batch.fps_t
                pred,loss1= model(padded_smiles_batch, batch, fps_t)
                total_preds1 = torch.cat((total_preds1, pred.cpu()), 0)
                total_labels1 = torch.cat((total_labels1, labels.cpu()), 0)

            AC, F1, SN, SP, CCR, MCC = evaluate(total_labels1.unsqueeze(1), total_preds1)
            res70.append([AC, F1, SN, SP, CCR, MCC])

        test_results60 = pd.DataFrame(res60, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_results60.mean()
        print("60:")
        print(f"test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}")
        mean_results.append([r['AC'], r['F1'], r['SN'], r['SP'], r['CCR'],r['MCC']])
        test_mean_results = pd.DataFrame(mean_results, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_mean_results.mean()
        print(
            f"mean_test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}")
        test_mean_results.to_csv(f'{args.output}/10_test_results60.csv', index=False)

        test_results70 = pd.DataFrame(res70, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_results70.mean()
        print("70:")

        print(f"test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}")
        mean_results1.append([r['AC'], r['F1'], r['SN'], r['SP'], r['CCR'],r['MCC']])
        test_mean_results = pd.DataFrame(mean_results1, columns=['AC', 'F1', 'SN', 'SP','CCR','MCC'])
        r = test_mean_results.mean()
        print(
            f"mean_test_---AC:{r['AC']}---F1:{r['F1']}---SN:{r['SN']}---SP:{r['SP']}---CCR:{r['CCR']}---MCC:{r['MCC']}")
        test_mean_results.to_csv(f'{args.output}/10_test_results70.csv', index=False)


if __name__ == '__main__':
    args = config.parse()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    main(args)
