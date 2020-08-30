# GanDTI

This repo contains the code for our paper " GanDTI: a Multi-task Neural Network for Drug-Target Interaction Prediction" 

by Shuyu Wang*, Peng Shan

we report GanDTI, an end-to-end deep learning model for both interaction classification and binding affinity prediction tasks. This model employs the compound graph and protein sequence data. It only consists of a graph neural network, an attention module and a multiple-layer perceptron, yet outperforms the state-of-the art methods on the DUD-E, human, and bindingDB benchmark datasets. This demonstrates our refined model is highly effective and efficient for DTI prediction and provides a new strategy for performance improvement.

# Dependencies

* Python 3.7
* Pytorch
* numpy
* pickle
* RDKit
* sklearn
* CUDA

# Usage

1. to test human dataset: 
first
```
cd human
```
and then run
```
python dataProcess.py 
```
to generate the files required. Then
```
cd ..
python main.py --dataset human --mode classification
```

2. to test BindingDB Ki dataset: 
first
```
cd ki
```
and then run
```
python dataProcess.py 
```
to generate the files required. Then
```
cd ..
python main.py --dataset ki --mode regression
```

3. to use DUD-E dataset: 
first
```
cd dude
```
and then unzip the dataset file to process
```
python dataProcess.py 
```
to generate the files required. Then
```
cd ..
python main.py --dataset dude --mode classification
```

# Acknowledgement
We'd like to express our gratitude towards all the colleagues and reviewers for helping us improve the paper. The project is impossible to finish without the open-source implementation.
