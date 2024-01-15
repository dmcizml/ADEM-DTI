# Required Packages

- python 3.8.13
- pytorch 1.7.0
- dgl 0.7.2
- cudatoolkit 11.0
- numpy 1.22.3
- requests 2.31.0
- scikit-learn 1.3.2
- scipy 1.6.2



## Data description

- `drug.txt` : list of drug names.
- `protein.txt` : list of protein names.
- `disease.txt` : list of disease names.
- `se.txt` : list of side effect names.
- `mat_drug_drug.txt` : Drug-Drug interaction matrix.
- `mat_drug_se.txt` : Drug-SideEffect association matrix.
- `mat_drug_disease.txt` : Drug-Disease association matrix.
- `mat_protein_protein.txt` : Protein-Protein interaction matrix.
- `mat_protein_disease.txt` : Protein-Disease association matrix.
- `mat_protein_drug.txt` : Protein-Drug interaction matrix.
- `mat_drug_protein.txt` : Drug-Protein interaction matrix.
- `Similarity_Matrix_Drugs.txt` : Drug similarity scores based on chemical structures of drugs
- `Similarity_Matrix_Proteins.txt` : Protein similarity scores based on primary sequences of proteins
- `mat_drug_protein_homo_protein_drug.txt` : Drug-Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper).
- `mat_drug_protein_drug.txt` : Drug-Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper).
- `mat_drug_protein_disease.txt` : Drug-Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper).



# Quick start

- python main.py


