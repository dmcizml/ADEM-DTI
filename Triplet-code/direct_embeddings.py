# requirements
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle 
from scipy import spatial

def save_direct_embeddings(path):

  # reading data
  drugFiles = [
      path+ '/mat_drug_drug.txt',
      path+ '/mat_drug_disease.txt',
      path+ '/mat_drug_se.txt',
      path+ '/Similarity_Matrix_Drugs.txt']

  proteinFiles = [
      path+ '/mat_protein_protein.txt',
      path+ '/mat_protein_disease.txt',
      path+ '/Similarity_Matrix_Proteins.txt',
  ]
  
  drug_drug = np.loadtxt(drugFiles[0])
  drug_disease = np.loadtxt(drugFiles[1])
  drug_se = np.loadtxt(drugFiles[2])
  Similarity_matrix_drug = np.loadtxt(drugFiles[3])
  
  protein_protein = np.loadtxt(proteinFiles[0])
  protein_disease = np.loadtxt(proteinFiles[1])
  Similarity_Matrix_Proteins = np.loadtxt(proteinFiles[2])

  # function to find similartites 
  def find_similarity_matrix(input_matrix):
    input_matrix_len = len(input_matrix)
    similarity_matrix = np.zeros((input_matrix_len, input_matrix_len))

    for i in range(input_matrix_len):
      for j in range(input_matrix_len):
        if np.all(input_matrix[i]==0) or np.all(input_matrix[j]==0):
          i_j_rows_similarity=0
        else:
          i_j_rows_similarity = 1 - spatial.distance.cosine(input_matrix[i], input_matrix[j])
        similarity_matrix[i][j] = i_j_rows_similarity

    return similarity_matrix


  # function for mixing different similarity matrixes
  def mix_matrix(input_matrixes, matrix_size, operation_name):

    number_of_input_matrixes = len(input_matrixes)

    summed_matrix = np.zeros((matrix_size, matrix_size))
    max_matrix = np.zeros((matrix_size, matrix_size))
    min_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
      for k in range(matrix_size):
        
        temp_sum = 0
        temp_max = 0
        temp_min = 0

        if operation_name == 'sum':
          for j in range(number_of_input_matrixes):
            temp_sum = temp_sum + input_matrixes[j][i][k]
            summed_matrix[i][k] = temp_sum
          
        if operation_name == 'max':
          for j in range(number_of_input_matrixes):
            temp_max = np.maximum(temp_max, input_matrixes[j][i][k])
            max_matrix[i][k] = temp_max

        if operation_name == 'min':
          for j in range(number_of_input_matrixes):
            temp_min = np.minimum(temp_max, input_matrixes[j][i][k])
            min_matrix[i][k] = temp_min     

    if operation_name == 'sum':
      return summed_matrix
    if operation_name == 'max':
      return max_matrix
    if operation_name == 'min':
      return min_matrix


  # find similarities for drugs
  # similarity between drug and se, protein, drug, and di 
  similarity_matrix_drug_se = find_similarity_matrix(drug_se)
  similarity_matrix_drug_disease = find_similarity_matrix(drug_disease)

  # get direct embeddings of drugs and save it
  mixed_drugs = mix_matrix([similarity_matrix_drug_se, similarity_matrix_drug_disease, Similarity_matrix_drug, drug_drug], drug_drug.shape[0], 'sum')
  np.savetxt(fname=path+ '/triplet_feature/mixed_drug_se_disease_drug_4matrix_708size.txt' , X = mixed_drugs)

  # find similarities for proteins 
  # similarity between protein and disease 
  similarity_matrix_protein_disease = find_similarity_matrix(protein_disease)

  # get direct embeddings of proteins and save it
  mixed_proteins = mix_matrix([similarity_matrix_protein_disease, protein_protein, Similarity_Matrix_Proteins], protein_protein.shape[0] , 'sum')
  np.savetxt(fname=path+ '/triplet_feature/mixed_protein_disease_protein_3matrix_1512size.txt' , X = mixed_proteins)
  
  
  similarity_matrix_disease_drug = find_similarity_matrix(drug_disease.T)
  similarity_matrix_disease_protein = find_similarity_matrix(protein_disease.T)
  mixed_diseases = mix_matrix([similarity_matrix_disease_drug, similarity_matrix_disease_protein], drug_disease.shape[1], 'sum')
  np.savetxt(fname=path+ '/triplet_feature/mixed_disease_drugprotein_2matrix_5603size.txt' , X = mixed_diseases)
  
  similarity_matrix_se_drug = find_similarity_matrix(drug_se.T)
  mixed_sideeffects = mix_matrix([similarity_matrix_se_drug], drug_se.shape[1], 'sum')
  np.savetxt(fname=path+ '/triplet_feature/mixed_sideeffect_1matrix_4192size.txt' , X = mixed_sideeffects)
  

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser() #description="training data")
  parser.add_argument('--data_path', type=str, default='../data/data_luo')

  args = parser.parse_args()
  config = vars(args)
  print(config)

  save_direct_embeddings(args.data_path)
  print('Done!')