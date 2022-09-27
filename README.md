# Fair Inference for Discrete Latent Variable Models

This folder provides all the source code and data to reproduce the experimental results as reported in [Islam, Rashidul, Shimei Pan, and James R. Foulds. "Fair Inference for Discrete Latent Variable Models." arXiv preprint arXiv:2209.07044 (2022)](https://arxiv.org/pdf/2209.07044.pdf). The code is tested on windows and linux operating systems. It should work on any other platform.

## Prerequisites

* Python
* PyTorch

Note that commonly used other Python libraries like NumPy, pandas, scikit-learn, etc. are also required. 

## Instructions

We provide three sperate folders "NB", "GMM", and "Special_Purpose_COMPAS" which contain Adult, HHP, and COMPAS datasets, respectively.
  
* Navigate to "NB" folder to run Vanilla-NB, DF-NB, GS-VAE, and GS-VFAE models on Adult dataset using the following script: "run_Vanilla_NB.py", "run_DF_NB.py", "run_GS_VAE.py", and "run_GS_VFAE.py", respectively.   
* Navigate to "GMM" folder to run Vanilla-GMM, DF-GMM, GS-VAE, and GS-VFAE models on HHP dataset using the following script: "run_Vanilla_GMM.py", "run_DF_GMM.py", "run_GS_VAE.py", and "run_GS_VFAE.py", respectively. 
* Navigate to "Special_Purpose_COMPAS" folder to run Vanilla-SP, and DF-SP models on COMPAS dataset using the following script: "run_Vanilla_SP.py", and "run_DF_SP.py", respectively. 

Note that these scripts will conduct the required hyper-parameters tuning on dev set first, and then save trained models and results on test set into "trained-models" and "results" folders, respectively.   

## Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.

