## LEO
This repository contains the source code and data samples for reproducing the experiments of our LEO method in our paper "LEO: An Information-theoretic Learning-based Approach for Identifying Out-of-distribution Source Code in Software Systems".

## Dependencies
We implemented our LEO method in Python using Tensorflow (version 2.9.0) and Python (version 3.9). Other required packages are transformers, keras, scikit-learn, numpy, scipy, and pickle.

## Datasets and Pre-trained models
The "data" folder contains source code samples (associated with specific CWE categories) used in the training and testing (inference) phases. The "leo" folder houses pre-trained models of our LEO method on the source code samples. Please download these folders including all of their files at [https://drive.google.com/drive/folders/1ml9-qZS76RJbMdoBANq9KiJvyHUYfhr9?usp=sharing.](https://drive.google.com/drive/folders/1iCmNv2oDtw4XFbgAntGIAwNcnea-gCE0?usp=sharing).

## Running LEO (the training process)
To train our LEO method, run the following command, for example:<br/>
*python leo.py --train --in_data=863 --lr=0.001 --lam=0.001 --tau=0.5 --tem=0.5 --cls=7*

Note that: in the above command, we use sample values for the hyperparameters. To change the value of the hyperparameters such as the learning rate, the batch size, and the trade-off, please change the corresponding arguments (refer to the *__main__* function in leo.py for details).

## Running LEO (the testing process)
To obtain the values for the metrics (i.e., FPR (at TPR 95%), AUROC, and AUPR) used in our paper, run the following command, for example: <br/>
*python leo.py --in_data=863 --ood_data=287 --lr=0.001 --lam=0.001 --tau=0.5 --tem=0.5 --cls=7*

Note that: please set the values for the hyperparameters as used in the training process. That will be useful for locating the corresponding folder where the model is saved from the training process.
