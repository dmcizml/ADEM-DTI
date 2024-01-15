import dgl
import numpy as np
import random
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score,  precision_recall_curve, accuracy_score
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import auc as auc3

# set random seed
def set_random_seed(seed):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# get the row, column and value of non-zero elements from a matrix
def get_nonzero_elements(matrix):
    # Numpy array
    if isinstance(matrix, np.ndarray):
        rows, cols = np.nonzero(matrix)
        values = matrix[rows, cols]
    # NumPy matrix
    elif isinstance(matrix, np.matrix):
        matrix = np.asarray(matrix)
        rows, cols = np.nonzero(matrix)
        values = matrix[rows, cols]
    else:
        raise ValueError("The input matrix type should be numpy.ndarray or scipy.sparse matrix")

    nonzero_elements = (rows, cols, values)
    return nonzero_elements

def load_data_dataset(network_path):
    
    # DTI
    drug_drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')
    
    # the path of drugs
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    disease_drug = drug_disease.T
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')
    sideeffect_drug = drug_sideeffect.T

    # the path of proteins
    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_protein_drug = drug_drug_protein.T
    protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')
    disease_protein = protein_disease.T
    
    # triplet feature
    protein_representations = np.loadtxt(network_path + 'triplet_feature/protein_triplet_representations.txt')
    drug_representations = np.loadtxt(network_path + 'triplet_feature/drug_triplet_representations.txt')
    disease_representations = np.loadtxt(network_path + 'triplet_feature/disease_triplet_representations.txt')
    sideeffect_representations = np.loadtxt(network_path + 'triplet_feature/sideeffect_triplet_representations.txt')
    
    # convert feature numpy to tensor
    drug_triplet = torch.from_numpy(drug_representations).to(torch.float32)
    protein_triplet = torch.from_numpy(protein_representations).to(torch.float32)
    disease_triplet = torch.from_numpy(disease_representations).to(torch.float32)
    sideeffect_triplet = torch.from_numpy(sideeffect_representations).to(torch.float32)

    # number of different nodes
    num_drug = drug_drug.shape[0]
    num_disease = disease_protein.shape[0]
    num_protein = protein_protein.shape[0]
    num_sideeffect = sideeffect_drug.shape[0]
    
    # get non-zero elements from array
    # drug
    drug_drug = get_nonzero_elements(drug_drug)
    drug_chemical = get_nonzero_elements(drug_chemical)
    drug_disease = get_nonzero_elements(drug_disease)
    disease_drug = get_nonzero_elements(disease_drug)
    drug_sideeffect = get_nonzero_elements(drug_sideeffect)
    sideeffect_drug = get_nonzero_elements(sideeffect_drug)
    drug_drug_protein = get_nonzero_elements(drug_drug_protein)

    # protein
    protein_protein = get_nonzero_elements(protein_protein)
    protein_protein_drug = get_nonzero_elements(protein_protein_drug)
    protein_sequence = get_nonzero_elements(protein_sequence)
    protein_disease = get_nonzero_elements(protein_disease)
    disease_protein = get_nonzero_elements(disease_protein)

    
    # Create dgl heterogeneous graph
    graph = dgl.heterograph({
        ('drug', 'ddp', 'protein'): (drug_drug_protein[0], drug_drug_protein[1]),
        ('protein', 'pdd', 'drug'): (protein_protein_drug[0], protein_protein_drug[1]),
        
        ('drug', 'dsimilarity', 'drug'): (drug_drug[0], drug_drug[1]),
        ('drug', 'chemical', 'drug'): (drug_chemical[0], drug_chemical[1]),
        ('drug', 'dse', 'sideeffect'): (drug_sideeffect[0], drug_sideeffect[1]),
        ('sideeffect', 'sed', 'drug'): (sideeffect_drug[0], sideeffect_drug[1]),
        ('drug', 'ddi', 'disease'): (drug_disease[0], drug_disease[1]),
        ('disease', 'did', 'drug'): (disease_drug[0], disease_drug[1]),
        
        ('protein', 'psimilarity', 'protein'): (protein_protein[0], protein_protein[1]),
        ('protein', 'sequence', 'protein'): (protein_sequence[0], protein_sequence[1]),
        ('protein', 'pdi', 'disease'): (protein_disease[0], protein_disease[1]),
        ('disease', 'dip', 'protein'): (disease_protein[0], disease_protein[1]),
    }, num_nodes_dict={'drug': num_drug, 
                       'disease': num_disease, 
                       'sideeffect': num_sideeffect, 
                       'protein': num_protein})

    # load dti
    dti_o = np.loadtxt(network_path + 'mat_drug_protein.txt')
    whole_positive_index = []
    whole_negative_index = []

    # get positive and negative samples
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            else:
                whole_negative_index.append([i, j])

    # set the ratio of positive and negative samples to 1:1
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size= len(whole_positive_index),
                                             replace=False)

    # create the dataset
    data_set = np.zeros((len(negative_sample_index) +  len(whole_positive_index), 3),
                        dtype=int)
    count = 0

    # save positive samples
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    
    # save negative samples
    for i in range(len(negative_sample_index)):
        data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
        data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
        data_set[count][2] = 0
        count += 1
    
    dateset = data_set
    
    # save edges of DPP graph
    DP_pairs = []

    # for a drug-protein pair, if there is the same drug or protein, there is an edge that connects it
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):
            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:  
                DP_pairs.append((i, j))
    
    # convert list to array 
    DP_pairs = np.array(DP_pairs)
    node_num = [num_drug, num_protein, num_disease, num_sideeffect]
    triplet = [drug_triplet, protein_triplet, disease_triplet, sideeffect_triplet]
    
    return dateset, graph, node_num, triplet, DP_pairs


# Create DPP graph, return graph and feature
def constructure_graph(dataset, dataedge, h1, h2):
    # cat protein and drug feature
    feature = torch.cat((h1[dataset[:, :1]], h2[dataset[:, 1:2]]), dim=2)
    # remove second dimension
    feature = feature.squeeze(1)
    
    # Create DPP graph
    graph = dgl.graph((dataedge[:, 0], dataedge[:, 1]), num_nodes=dataset.shape[0])

    return graph, feature


# divide the dataset and perform cross validation
def get_cross(data, split):
    # set1: training set for each fold
    # set2: testing set for each fold
    set1 = []
    set2 = []
    skf = KFold(n_splits=split, shuffle=True)
    # skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1, set2

# calculate AUC
def get_roc(out, label):
    return roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy())

# calculate AUPR
def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)

# calculate F1
def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())

# calculate precision
def get_precisionscore(out, label):
    return precision_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())

# calculate recall
def get_recallscore(out, label):
    return recall_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())

# calculate MCC
def get_mccscore(out, label):
    return matthews_corrcoef(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())

# calculate ACC
def get_accscore(out, label):
    return accuracy_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())

# calculate L2 regularization
def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg

# load data
def load_dataset(dateName):
    if dateName == "data_luo":
        network_path = "./data/data_luo/"
    elif dateName == "data_zeng":
        network_path = "./data/data_zeng/"
    elif dateName == "data_li":
        network_path = "./data/data_li/"
    return load_data_dataset(network_path)
