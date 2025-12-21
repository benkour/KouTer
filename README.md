# A green solvent screening tool for emerging materials via uncertainty aware, transformer enhanced  transfer learning


This repository includes the code base of the paper **"A green solvent screening tool for emerging materials via uncertainty aware, transformer enhanced  transfer learning"**. 

## Table of Contents

1. [Summary](#summary)
2. [Prerequisites ](#prerequisites)
3. [Parameter and Input Specification](#parameter-specification)
4. [How to Run](#how_to_run)




## Summary <a name="summary"></a>

Our pipeline consists of
- A customly altered pre-trained foundational model (DimeNet) that outputs a vector for each molecule
- A transformer model
- A Gaussian processes model
- Optional Chemical Description 


All options are configurable in the parameters.json file

## Prerequisites<a name="prerequisites"></a>

**Tested Configuration**:

- Operating System: Ubuntu 22.04 (jammy)
- IDE: Visual Studio Code 

**Note**: As a rule, everything should work on other systems as well but we cant guarantee it.

-The data can be downloaded from:

https://syncandshare.lrz.de/getlink/fiUHBk5ULNmtNTfL3ZBqMV/

and need to be placed in the **data** folder.

- The specific library versions we used with can be found in the requirements.txt. They can be installed directly using `pip install -r requirements.txt` from the terminal on the folder level that the requirements.txt is.
The code will also benefit from CUDA should you have an NVIDIA graphics card





## Parameter configuration <a name="parameter-specification"></a>

The pipeline can be deeply altered by diving into the source code. Nevertheless, a user friendly approach to change 
the model architecture and training is through altering the **parameters.json** file.


**parameters.json parameters**
| Parameter                     | Type    | Description                                                                                                                                                                                                                                                                                |
| ----------------------------- | ------  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `prediction_target`              | List    |a list of strings containing the names of the properties you wish to create predictions for (as named in the data folder)                                                                            |
| `dimenet_target`            | List    | a list of properties used to pretrained DimeNet to be used. Can take any subset (excluding the empty set) of the values [0, 1,2, 5, 6, 7, 8, 9, 10, 11]                                                                                                                                                                                                                   |
| `mean_module`           | List  | a list of gaussian processes mean modules to be used. Currently supported : "LinearMean", ConstantMean", "ZeroMean"
|
| `standardization`         | List    | a list of boolean values that determines whether to standardize the values of the targets or not.                                                                                                                                                                                                              |
| `uncertainty`         | List    | a list of boolean values that determines whether to use Gaussian Processes as the final layer of the model or not. It is a list because it can be sweeped as any other hyperparameter.                                                                                                                                                                                                                       |
| `log_flag`                 | Bool    |  true or false depending on if the values of the target should be processed by a log10 preprocessing function. Useful when the target spans many orders of magnitude                                                                                                                                                                                                                             |
| `transfomers_flag`                | Bool    | rue or false depending on whether a transformer layer should be used between the foundational model and the Gaussian processes                                                                                                                                                                                                                                     |
| `nr_of_mixtures`             | Integer  | Currently the Gaussian Processes Kernel function is fixed to Gaussian Mixtures. This value determines how many mixtures this kernel should have. Can only be an integer                                                                                                                                                                                            |
| `descriptor_flag`     | Bool    | true or false depending on whether you want to use molecular descriptors along with the foundational model outputs                                                                                 |
| `save_folder`       | String    | a string denoting the folder where the model results will be saved. If the folder doesnt exist it will be automatically generated                                                                                                                                                               |
| `validation`                  | Bool     | true or false depending on whether you want to use a validation set for early stopping. Useful against overfitting in large datasets, detrimental in very small ones.                                                                                                                                                               |
| `normalize_descriptors`      | Bool   |  true or false denoting if the descriptors will be normalized or not                                                                                                                                                                                      |
| `transformer_heads`    | Integer    | an integer denoting the number of the transformer heads. -1 will use as many heads as the feature dimensionality of the transformer layer input                                                                                                        |
| `solvent_prediction_flag` | Bool    | true or false depending on whether you want to perform predictions on the large VOC dataset                                                                                                     |
| `dimenet_flag`            | Bool  | rue or false depending on whether you want to use the pretrained foundational model or not                                                                                                                                                                                                          |
| `keep_all_descriptors`                 | Bool     | true or false depending on whether you want to keep all descriptors calculated.                                                                                                                                                                                                                                     |
| `output_dim`                      | Integer     | a value denoting the dimensionality of the features entering the trainable part of the pipeline Only relevant if `keep_all_descriptors` is false. It discards every descriptor beyond the number given, -7 if dimenet is used. i.e. if dimenet is used, there are 30 descriptors and this value is 17 the last 20 descriptors will be discarded                                                                                                                                                                                                                                       |
| `keep_all_descriptors`          | Bool    | true or false depending on whether you want to keep all descriptors calculated |
| `EPOCHS`    | Integer    | number of training epochs                                                                                                                                                                     |
| `nr_of_runs`             | Integer     | number of folds in the k-fold validation                                                                                                                                                          |

 ## How to Run <a name="how_to_run"></a>
 - Open the terminal and navigate to the folder level that contains the *main.py* file. 
 - Activate the environment with the libraries if you have created one. If you have installed the requirements on the default Path, ignore this step. 
 - Type python main.py and press enter


