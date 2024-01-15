# requirements
import argparse
import sys
import io
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
#import tensorflow_datasets as tfds
from sklearn.cluster import KMeans

# kmeans function where k is the number of chosen clusters
def k_means(X, k):
  kmeans = KMeans(n_clusters= k, random_state=0).fit(X)
  predicted_labels = kmeans.labels_
  return kmeans, predicted_labels

# function to see how many items are inside each cluster
def check_counts_in_clusters(kmeans_predicted_labels, k):
  counted_list = [ np.count_nonzero(kmeans_predicted_labels == i) for i in range(k)]
  return counted_list

# this function prints out information about clusters for choosing k between 2 and 66
def decide_about_threshold(data):

  different_clusterings_counts_list = {}
  for i in range(2, 66):
    kmeans_i , predicted_labels_ki = k_means(data, i)
    count_clusters_list = check_counts_in_clusters(predicted_labels_ki, i)
    print(count_clusters_list)
    different_clusterings_counts_list[i] =  count_clusters_list
    
    for i in different_clusterings_counts_list.keys():
      minimum = np.min(different_clusterings_counts_list[i])
      maximum = np.max(different_clusterings_counts_list[i]) 
      difference = maximum - minimum
      average = np.mean(different_clusterings_counts_list[i]) 
      print('Number of clusters is:', i)
      print('min of list: ' ,minimum , ' max of list: ', maximum ,  'mean of list: ', average , ' difference: ', difference , '\n')


def triplet_model(X,y, Input_Shape):
  
  model = tf.keras.Sequential([
      tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(Input_Shape, 1)),
      #tf.keras.layers.MaxPooling2D(pool_size=2),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'),
      #tf.keras.layers.MaxPooling2D(pool_size=2),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
      tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
  ])

  # Compile the model
  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=tfa.losses.TripletSemiHardLoss())

  # Train the network
  history = model.fit(
      X,y,
      epochs=5,
      batch_size = 32,
      shuffle=True)
  
  return model

def apply_triplet_on_drugs(path, k):
  mixed_drugs = np.loadtxt(path+ "/mixed_drug_se_disease_drug_4matrix_708size.txt") 

  kmeans_k , predicted_labels_k = k_means(mixed_drugs, k)
  X = mixed_drugs.reshape(-1, mixed_drugs.shape[1], 1)
  y = predicted_labels_k
  
  #train model with triplet loss for obtaining drug embeddings
  Input_Shape = mixed_drugs.shape[1]
  model_d = triplet_model(X, y, Input_Shape)

  # save model 
  # model_d.save('drug__triplet__model.h5')

  # save representation
  drug_representations = model_d.predict(X)
  print(drug_representations.shape)
  np.savetxt(fname=path+ '/drug_triplet_representations.txt' , X = drug_representations)

def apply_triplet_on_proteins(path, k):
  mixed_proteins = np.loadtxt(path+ "/mixed_protein_disease_protein_3matrix_1512size.txt") 
  kmeans_k , predicted_labels_k = k_means(mixed_proteins, k)
  X = mixed_proteins.reshape(-1, mixed_proteins.shape[1], 1)
  print(X.shape)
  
  y = predicted_labels_k

  #train model with triplet loss for obtaining protein embeddings
  Input_Shape = mixed_proteins.shape[1]
  print(Input_Shape)
  model_p = triplet_model(X, y, Input_Shape)
  
  # save model 
  # model_p.save('protein__triplet__model.h5')

  # save representation
  protein_representations = model_p.predict(X)
  print(protein_representations.shape)
  np.savetxt(fname= path+ '/protein_triplet_representations.txt' , X = protein_representations)


def apply_triplet_on_diseases(path, k):
  mixed_diseases = np.loadtxt(path+ "/mixed_disease_drugprotein_2matrix_5603size.txt") 

  kmeans_k , predicted_labels_k = k_means(mixed_diseases, k)
  X = mixed_diseases.reshape(-1, mixed_diseases.shape[1], 1)
  y = predicted_labels_k
  #train model with triplet loss for obtaining drug embeddings
  Input_Shape = mixed_diseases.shape[1]
  model_di = triplet_model(X, y, Input_Shape)

  # save model 
  # model_d.save('drug__triplet__model.h5')

  # save representation
  disease_representations = model_di.predict(X)
  print(disease_representations.shape)
  np.savetxt(fname=path+ '/disease_triplet_representations.txt' , X = disease_representations)

def apply_triplet_on_sideeffects(path, k):
  mixed_sideeffects = np.loadtxt(path+ "/mixed_sideeffect_1matrix_4192size.txt") 

  kmeans_k , predicted_labels_k = k_means(mixed_sideeffects, k)
  X = mixed_sideeffects.reshape(-1, mixed_sideeffects.shape[1], 1)
  y = predicted_labels_k
  
  #train model with triplet loss for obtaining drug embeddings
  Input_Shape = mixed_sideeffects.shape[1]
  model_se = triplet_model(X, y, Input_Shape)

  # save model 
  # model_d.save('drug__triplet__model.h5')

  # save representation
  sideeffect_representations = model_se.predict(X)
  print(sideeffect_representations.shape)
  np.savetxt(fname=path+ '/sideeffect_triplet_representations.txt' , X = sideeffect_representations)


if __name__ == "__main__":

  parser = argparse.ArgumentParser() #description="training data")
  parser.add_argument('--data_path', type=str, default='./data/data_luo/triplet_feature')
  parser.add_argument('--num_of_drug_clusters', type= int, default=4)
  parser.add_argument('--num_of_protein_clusters', type= int, default=5)
  parser.add_argument('--num_of_disease_clusters', type= int, default=6)
  parser.add_argument('--num_of_sideeffect_clusters', type= int, default=8)
  parser.add_argument('--find_best_k', type= bool, default= False)

  args = parser.parse_args()
  config = vars(args)
  print(config)

  if args.find_best_k== True:
    print('for drugs:')
    mixed_drugs = np.loadtxt(args.data_path+ "/mixed_drug_se_disease_drug_4matrix_708size.txt") 
    decide_about_threshold(mixed_drugs)
    
    print('for protein:')
    mixed_proteins = np.loadtxt(args.data_path+ "/mixed_protein_disease_protein_3matrix_1512size.txt") 
    decide_about_threshold(mixed_proteins)
    
    print('for disease:')
    mixed_diseases = np.loadtxt(args.data_path+ "/mixed_disease_drugprotein_2matrix_5603size.txt") 
    decide_about_threshold(mixed_diseases)

    print('for sideeffect:')
    mixed_sideeffects = np.loadtxt(args.data_path+ "/mixed_sideeffect_1matrix_4192size.txt") 
    decide_about_threshold(mixed_sideeffects)
    
  print("Obtaining drug representations...")
  apply_triplet_on_drugs(args.data_path, args.num_of_drug_clusters)

  print("Obtaining protein representations...")
  apply_triplet_on_proteins(args.data_path, args.num_of_protein_clusters)
  
  print("Obtaining disease representations...")
  apply_triplet_on_diseases(args.data_path, args.num_of_disease_clusters)

  print("Obtaining sideeffect representations...")
  apply_triplet_on_sideeffects(args.data_path, args.num_of_sideeffect_clusters)

