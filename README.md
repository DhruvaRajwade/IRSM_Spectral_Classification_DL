# IRSM Spectral Classification using Machine Learning and Deep Learning
This repository contains code used for our work to predict idiopathic recurrent spontaneous miscarriages using Deep Learning (Specifically Variational Autoencoders, Artificial Neural networks and Convolutional Neural Networks (1D) as well as Classical Machine Learning models)

### To run the code on your setup:
1. Clone the repository to your local machine:
 ```python
 git clone https://github.com/DhruvaRajwade/IRSM_Spectral_Classification_DL.git
 ```

2. Install the required dependencies:
```python
pip install -r requirements.txt
```
## Files

```
|___Root
|
├── R_code/ # Requires the Chemospec package 
│   ├── HCA.R # Sample R Code for Hierarchical Clustering using Agglomerate Euclidean distances
│   └── LDA.R # Sample R Code for a Linear discriminant Analysis
│
├── dev/  # Some experimental code I implemented (Not included in the paper, but interesting nevertheless)
│   ├── PINN.py 
│   └── forward_forward_NIPS.py
│
├── sklearn/ # Classical Machine Learning code
│   ├── cross_validation.py # Functions for Cross Validation and Visualization
│   ├── grid_search.py      # Functions to conduct a GridSearch Hyperparameter sweep
│   ├── model_selection.py  # Functions to help in initial model selection
│   ├── pca_visualize.py    # Visualize PCA outputs using Loadings and scatter plots
│   └── plot.py             # Helper script for grid_search.py
│
├── tensorflow/
│   ├── cnn_bayesian_hyperparam_tuning.py # Bayesian hyperparameter tuning using keras-tuner
│   ├── tf_cnn.py # 1 D CNN
│   └── tf_eval.py # Evaluation and model inference
│
├── torch/
│   ├── torch_eval.py # Evaluation and model inference
│   ├── torch_models.py # Variational Autoencoder and Attention Residual ANN
│   ├── torch_train.py  # Training script
│   ├── vae_cross_validation.py # Cross-validate the pipeline
│   └── vae_loss.py    # Loss functions {KL Divergence and BCE hybrid loss controlled by `Temperature` param, Negative Log Likelihood Loss and MSE loss implementations}
│
├── utils/
│   └── data_preprocessing.py # Helper functions to convert the data from raw form to Numpy (for Sklearn and TF) or Torch tensors(For Pytorch)
│
├── README.md (What you're reading)
└── requirements.txt (Generated using pipreqs!)
```

### Todo: Add Results and sample visualizations, add Example Notebooks , Add RayTune hyperparameter sweep code
