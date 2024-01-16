import argparse
from utils import *
from model import *
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)    


def train(model, args, optim, train_index, graph, dtidata, label, node_feature, dpedge):
    model.train()
    # forward  
    out, d, p = model(graph, node_feature, train_index, dtidata, dpedge)
    # L2 regularization term
    reg = get_L2reg(model.parameters())
    # loss
    loss = F.nll_loss(out, label[train_index].reshape(-1)) + args.reg_loss_co * reg
    # autograd
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item(), d, p


def evaluate(model, d, p, test_index, graph, dtidata, label, node_feature, dpedge):
    model.eval()
    with torch.no_grad():  
        # forward
        out = model(graph, node_feature, test_index, dtidata, dpedge, iftrain=False, d=d, p=p)
        # evaluating metrics
        test_acc = get_accscore(out, label[test_index])
        test_auc = get_roc(out, label[test_index])
        test_aupr = get_pr(out,label[test_index])      
        test_f1 = get_f1score(out,label[test_index])     
        test_precision = get_precisionscore(out,label[test_index])    
        test_recall = get_recallscore(out,label[test_index])    
        test_mcc = get_mccscore(out,label[test_index])
    return test_acc, test_auc, test_aupr, test_f1, test_precision, test_recall, test_mcc

def run_model(args):
    # load dataset
    dtidata, graph, num, triplet, dpedge = load_dataset(args.dataset)

    graph = graph.to(args.device)
    label = torch.tensor(dtidata[:, 2:3]).to(args.device)
    
    # feature initialization strategy
    if args.init_type == 'triplet':
        hdd = triplet[0].to(args.device)
        hdp = triplet[1].to(args.device)
        hdi = triplet[2].to(args.device)
        hds = triplet[3].to(args.device)
    elif args.init_type == 'random':
        hdd = torch.randn((num[0], args.in_size)).to(args.device)
        hdp = torch.randn((num[1], args.in_size)).to(args.device)
        hdi = torch.randn((num[2], args.in_size)).to(args.device)
        hds = torch.randn((num[3], args.in_size)).to(args.device)
    # input features
    node_feature = {'drug':hdd, 'protein':hdp, 'disease':hdi, 'sideeffect':hds}

    # save results
    f_csv = open(args.save_dir + 'results.csv', 'a')
    f_csv.write('Fold,ACC,AUC,AUPR,F1,PRECISION,RECALL,MCC\n')
    f_csv.close()

    # 10-fold cross-validation
    train_indeces,test_indeces = get_cross(dtidata, args.nFold)
    
    all_acc = []
    all_auc = []
    all_f1 = []
    all_aupr = []
    all_precision = []
    all_recall = []
    all_mcc = []
    
    for i in range(len(train_indeces)): 
        # training set
        train_index = train_indeces[i]
        
        # testing set
        test_index = test_indeces[i]
        
        # network    
        model = ADEM(args = args).to(args.device)  
            
        optim = torch.optim.Adam(lr=args.lr, weight_decay=args.weight_decay, params=model.parameters())
        make_dir(args.save_dir + '/checkpoint')
        make_dir(args.save_dir + '/bestfeature')
        
        best_acc = 0
        best_f1 = 0
        best_aupr = 0
        best_auc = 0
        best_precision = 0
        best_recall = 0
        best_mcc = 0
        counter = 0
        best_weights = None
        best_drug = None
        best_protein = None
        
        # only_test
        if args.only_test:
            # load model parameters
            model.load_state_dict(torch.load(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(i), map_location=args.device))
            only_test_drug = torch.load(args.save_dir + '/bestfeature/bestdrug_fold_{}.txt'.format(i), map_location=args.device)
            only_test_protein = torch.load(args.save_dir + '/bestfeature/bestprotein_fold_{}.txt'.format(i), map_location=args.device)
            acc, auc, aupr, f1, precision, recall, mcc = evaluate(model, only_test_drug, only_test_protein, test_index, graph, dtidata, label, node_feature, dpedge)
            best_acc, best_auc, best_aupr, best_f1, best_precision, best_recall, best_mcc = acc, auc, aupr, f1, precision, recall, mcc 
        # Train and validation
        else:
            for epoch in range(args.epochs):
                # Train
                loss, d, p = train(model, args, optim, train_index, graph, dtidata, label, node_feature, dpedge)
                # Validation
                acc, auc, aupr, f1, precision, recall, mcc = evaluate(model, d, p, test_index, graph, dtidata, label, node_feature, dpedge)
                print(f"train loss is {loss:.4f}, acc is {acc:.4f}, auc is {auc:.4f}, aupr is {aupr:.4f}, f1 is {f1:.4f}, precision is {precision:.4f}, recall is {recall:.4f}, mcc is {mcc:.4f}")
    
                # best results
                if auc > best_auc:
                    best_acc = acc
                    best_aupr = aupr
                    best_f1 = f1
                    best_auc = auc
                    best_precision = precision
                    best_recall = recall
                    best_mcc = mcc
                    counter = 0
                    best_weights = model.state_dict()
                    best_drug = d
                    best_protein = p
                    
                else:
                    # Early stopping
                    if args.EarlyStopping == True:
                        counter = counter + 1
                        if counter > args.patience:
                            print('Early stopping!')
                            break   
                    # No early stopping
                    else:
                        continue
            # save best model parameters
            torch.save(best_weights, args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(i))
            torch.save(best_drug, args.save_dir + '/bestfeature/bestdrug_fold_{}.txt'.format(i))
            torch.save(best_protein, args.save_dir + '/bestfeature/bestprotein_fold_{}.txt'.format(i))
            
        all_acc.append(best_acc)
        all_auc.append(best_auc)
        all_f1.append(best_f1)
        all_aupr.append(best_aupr)
        all_precision.append(best_precision)
        all_recall.append(best_recall)
        all_mcc.append(best_mcc)
        
        # save best results
        f_csv = open(args.save_dir + 'results.csv', 'a')
        f_csv.write(','.join(map(str, [i, best_acc, best_auc, best_aupr, best_f1, best_precision, best_recall, best_mcc])) + '\n')
        f_csv.close()
    
        print(f"fold{i} acc is {best_acc:.4f} f1 is {best_f1:.4f} auroc is {best_auc:.4f} aupr is {best_aupr:.4f} precision is {best_precision:.4f} recall is {best_recall:.4f} mcc is {best_mcc:.4f}")

    # calculate mean and std
    print(f"{args.dataset}, ave acc is {np.mean(all_acc):.4f}, ave auc is {np.mean(all_auc):.4f},ave aupr is {np.mean(all_aupr):.4f} ,ave f1 is {np.mean(all_f1):.4f},ave precision is {np.mean(all_precision):.4f},ave recall is {np.mean(all_recall):.4f},ave mcc is {np.mean(all_mcc):.4f}")
    print(f"{args.dataset}, ave acc is {np.std(all_acc):.4f}, ave auc is {np.std(all_auc):.4f},ave aupr is {np.std(all_aupr):.4f} ,ave f1 is {np.std(all_f1):.4f},ave precision is {np.std(all_precision):.4f},ave recall is {np.std(all_recall):.4f},ave mcc is {np.std(all_mcc):.4f}")

    # save mean and std
    f_csv = open(args.save_dir + 'results.csv', 'a')
    f_csv.write(','.join(map(str, ['mean', np.mean(all_acc), np.mean(all_auc), np.mean(all_aupr), np.mean(all_f1), np.mean(all_precision), np.mean(all_recall), np.mean(all_mcc)])) + '\n')
    f_csv.write(','.join(map(str, ['std', np.std(all_acc), np.std(all_auc), np.mean(all_aupr), np.mean(all_f1), np.mean(all_precision), np.mean(all_recall), np.mean(all_mcc)])) + '\n')
    f_csv.close()
    
def parser():
    ap = argparse.ArgumentParser(description='DTI testing for the recommendation dataset')
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--dataset', type=str, default='data_luo', help='Options: data_luo, data_qi, data_li.')
    ap.add_argument('--patience', type=int, default=100, help='Early stopping. Default is 100.')
    ap.add_argument('--num_layers', type=int, default=5, help='The number layers of model. Default is 5.')
    ap.add_argument('--in_size', type=int, default=256, help='Dimension of the input size. Default is 256.')
    ap.add_argument('--hidden_size', type=int, default=256, help='Dimension of the hidden size. Default is 256.')
    ap.add_argument('--out_size', type=int, default=128, help='Dimension of the out size. Default is 128.')
    ap.add_argument('--attn_size', type=int, default=256, help='Dimension of the attention aize. Default is 256.')
    ap.add_argument('--dpg_size', type=int, default=256, help='Dimension of the DPP graph node feature size. Default is 256(in fact is 512(drug and protein)).')
    ap.add_argument('--lr', type=float, default=0.0001, help='Number of lr. Default is 0.0001.')
    ap.add_argument('--weight_decay', type=float, default=5e-4, help='Number of weight_decay. Default is 5e-4.')
    ap.add_argument('--dropout', type=float, default=0.5, help='Number of dropout. Default is 0.5.')
    ap.add_argument('--epochs', type=int, default=1000, help='Number of epochs. Default is 1000.')
    ap.add_argument('--reg_loss_co', type=float, default=0.0007, help='Number of reg_loss_co. Default is 0.0007.')
    ap.add_argument('--nFold', type=int, default=10, help='Number of fold. Default is 10.')
    ap.add_argument('--init_type', type=str, default='triplet', help='Initialization strategy. Options: triplet, random.')
    ap.add_argument('--gnn_type', type=str, default='mixhop', help='The gnn type to deal with DPP graph. Options: mixhop, gcn, gat, sage.')
    ap.add_argument('--bias', type=bool, default=True, help='Default is True.')
    ap.add_argument('--batchnorm', type=bool, default=True, help='Default is True.')
    ap.add_argument('--only_test', type=bool, default=False, help='Default is False.')
    ap.add_argument('--EarlyStopping', type=bool, default=False, help='Default is False.')
    ap.add_argument('--predictor', type=str, default='dpg', help='What method is used for prediction. Default is dpg(DP graph). Options: dpg, lin')
    ap.add_argument('--save_dir', type=str, default='./results_dpp/{}/repeat{}/gnn_type_{}/', help='Postfix for the saved model and result.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    args = ap.parse_args()
    return args
    
if __name__ == '__main__':
    seed = 47
    # set seed
    set_random_seed(seed)
    
    args = parser()
    # node types 
    ntypes = ['drug', 'protein', 'disease', 'sideeffect']
    
    # edge types
    etypes = ['ddp', 'pdd', 'dsimilarity', 'chemical', 'dse', 'sed', 'ddi', 'did', 'psimilarity', 'sequence', 'pdi', 'dip']
    
    args.ntypes = ntypes
    args.etypes = etypes
    
    for rp in range(args.repeat):
        print('This is repeat ', rp)
        args.rp = rp
        # Save results directory
        args.save_dir = args.save_dir.format(args.dataset, args.rp, args.gnn_type)
        print('Save path ', args.save_dir)
        make_dir(args.save_dir)
        # logger
        sys.stdout = Logger(args.save_dir + 'log.txt')
        run_model(args)