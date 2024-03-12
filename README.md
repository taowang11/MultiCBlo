# MVML-MPI: Multi-View Multi-Label Learning for Metabolic Pathway Inference

## Overview
MVML-MPI: Multi-View Multi-Label Learning for Metabolic Pathway Inference. By incorporating distinct compound feature sets, corresponding compound encoders, and an attention mechanism providing a fusion module for feature refinement.
![image](./img/MVML-MPI.png)

## Dependencies
The package depends on the Python==3.7.15:
```
dgl==0.9.1
dgllife==0.3.0
numpy==1.21.6
pandas==1.3.5
torch==1.12.1
torch_geometric==2.2.0
torch_scatter==2.1.0
torch_sparse==0.6.15 
scikit-learn==0.24.2  
rdkit==2022.9.3
```

## Datasets
The dataset employed in this study was initially sourced from the publicly accessible KEGG pathway database, comprising 11 distinct types of metabolic pathways.

The data preprocessing was carried out following the strategy outlined in [Auto-MRS](https://github.com/AutoMachine0/Auto-MSR).

The statistics of the metabolic pathway dataset are shown below:

| Metabolic Pathway                  | No.of Samples | Ratio |
|---------------------------------------------|-------------------------|----------------|
| Biosynthesis of Other Secondary Metabolites | 1084                    | 25.80%         |
| Terpenoids and Polyketides Meatbolism       | 898                     | 21.40%         |
| Xenobiotics Biodegradation and Meatbolism   | 661                     | 15.80%         |
| Lipid Metabolism                            | 562                     | 13.40%         |
| Amino Acid Meatbolism                       | 416                     | 9.90%          |
| Cofactors and Vitamins                      | 407                     | 9.70%          |
| Carbohydrate Meatbolism                     | 253                     | 6.00%          |
| Nucleotide Meatbolism                       | 108                     | 2.60%          |
| Meatbolism of Other Amino Acids             | 103                     | 2.50%          |
| Energy Metabolism                           | 102                     | 2.40%          |
| Glycan Metabolism                           | 96                      | 2.30%          |


The datasets are stored in ```./data/kegg_dataset.csv``` and contains 4,192 compounds' SMILES and labels.

The file ```./data/data_index.txt``` contains the index numbers of train, validation, and test sets.

## Running the Experiment
To run our model based on the default conditions:
```bash
$ python main.py 
```
The following parameters can be used to customize our model:

<kbd>output</kbd> specifies the path to store the model. 

<kbd>hidden_feats</kbd> determines the output size of an attention head in the i-th GAT layer.

<kbd>rnn_embed_dim</kbd> sets the embedding size of each SMILES token.

<kbd>rnn_hidden_dim</kbd> defines the number of features in the RNN hidden state.

<kbd>fp_dim</kbd> specifies the hidden size of fingerprints module.

<kbd>head</kbd> denotes the head size of multi-view attention.


Use the command <code>python main.py -h</code>to check the meaning of other parameters.
