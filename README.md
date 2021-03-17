<div align="center"> 

# Dyslipidemia Prediction 
</div>

In this repository, we utilize machine learning and deep learning techniques to predict Apolipoprotein C-III (ApoC-III), which is the measurement affecting indirectly blood lipid disease. Concretely, TG and LDL-C are two main factors causing Dyslipidemia by clinging to the vessel wall obstructs blood circulation. Lipoprotein lipase (LPL) is an enzyme in other to hydrolysis TG while ApoC-III inhibits LPL. They operate and regulate the body continuously. Decreasing ApoC-III makes reducing TG and LDL-C so helps reduce subclinical atherosclerosis in the body. Conversely, it is harmful to the body by increasing TG in the blood. We incorporate Single-nucleotide polymorphism (SNP) of patients with treatments that are drugs that help reduce ApoC-III. Compatible metrics for each person are different. Therefore, we select the treatment which has the minimum ApoC-III score will be proposed as the main treatment method for different patients.

### Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Install](#3-install)
4. [Getting started](#4-getting-started)
5. [Repo structure](#5-repo-structure)
6. [Results](#6-results) 

### 1. Introduction
#### Desciption of Dataset
- The dataset is collected **384 SNP** which are extracted from **978 different patients**. 
- Each SNP contains one of four genotypes `AA, AB, BB, NC` (allen B is superior to allen A and NC is represented for unknown).
- There are three kinds of **treatments**: `𝑓𝑒𝑛𝑜𝑓𝑖𝑏𝑟𝑖𝑐 𝑎𝑐𝑖𝑑, 𝑓𝑒𝑛𝑜𝑓𝑖𝑏𝑟𝑖𝑐 𝑎𝑐𝑖𝑑 𝑠𝑡𝑎𝑡𝑖𝑛, 𝑠𝑡𝑎𝑡𝑖𝑛 𝑎𝑙𝑜𝑛𝑒`.
- Prediction value **outcome** is a real value which is represented for ApoC-III score.


#### Models
Regression models used in this project as below:
- **Machine Learning Models**: Linear Regression, Regularization L2 Linear Regression, Regularization L1 Linear Regression, ElasticNet, MultiTask ElasticNet, Support Vector Regression, Decision Tree Regression, XGBoost, AdaBoost, ExtraTreeRegressor, GradientBoostingRegressor, RandomForest, LightGBM.
- **Deep Learning Models**: Neural Network, Convolutional Neural Neural.


### 2. Prerequisites
- Python >= 3.6
- PyTorch >= 1.7
- Sklearn >= 0.23
- Other dependencies described in `requirements.txt`


### 3. Install
Create Conda virtual environment:

```bash
conda create -n dyslipidemia python=3.8 -y
conda activate dyslipidemia
pip install -r requirements.txt
```

### 4. Getting started
Run all models on public dataset
```
python main.py all --mode public
```
There are several control parameters: 
- **model**: Run mlmodel or dlmodel or all [mlmodel, dlmodel, all]
- **mode**: Choose dataset [public, private]

(And more parameters are configed in args)

### 5. Repo structure
- **dataset:** Dataset used in this repository
- **lib**
   - **models:**
      - **abstract_model.py:** Abstract class for models
      - **cnn_regression.py:** Convolutional Neural Network model
      - **dl_model.py:** Run deep learning model
      - **ml_model.py:** Run machine learning model
      - **nn_regression.py:** Neural Network model
  - **config.py:** Configuration
  - **runner.py:** Executes all models by mode
- **utils**
  - **load_data.py**: Loads dataset from dataset folder and transform by mode.
- **logging**: Contain log file folder
- ~~**plot_EDA.ipynb**: Explore Data Analysis and Plot char.~~ 
- **main.py:** Config and run the runner. 


### 6. Results 
The result of public and private datasets shows in the table below:

#### Machine Learning Models
|             Model Name              |       Split [80-20] (public)    | K-fold (public) |       Split [80-20] (private)    | K-fold (private) |
|                :---                 |              :---:              |      :---:      |               :---:              |      :---:       |
| Linear Regression                   | 25.045 | 24.672 | 25.503 | 29.723 |
| Regularization L2 Linear Regression | 24.962 | 24.591 | 25.403 | 29.552 |
| Regularization L1 Linear Regression | 23.791 | 23.421 | 23.581 | 27.320 |
| ElasticNet                          | 20.585 | 19.963 | 16.773 | 16.748 |
| MultiTask ElasticNet                | 22.425 | 22.064 | 21.207 | 24.094 |
| Support Vector Regression           | 20.948 | 20.141 | 17.234 | 17.208 |
| Decision Tree Regression            | 27.862 | 29.044 | 46.827 | 43.348 |
| XGBoost                             | 23.142 | 21.558 | 25.630 | 24.260 |
| AdaBoost                            | 21.265 | 20.337 | 16.922 | 17.409 |
| ExtraTreeRegressor                  | 21.223 | 20.223 | 19.532 | 19.603 |
| GradientBoostingRegressor           | 22.202 | 21.393 | 25.986 | 27.510 |
| RandomForest                        | 20.991 | 20.490 | 20.534 | 19.927 |
| LightGBM                            | 22.138 | 21.184 | 18.057 | 18.676 |

#### Deep Learning Models
|             Model Name              |       Split [80-20] (public)    |       Split [80-20] (private)    |
|                :---                 |              :---:              |               :---:              |
| Neural Network                      | 22.181 | 19.254 |  
| Convolutional Neural Neural         | 22.363 | 22.207 |
