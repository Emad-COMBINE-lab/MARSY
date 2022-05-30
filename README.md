# MARSY: A multitask deep learning framework for prediction of drug combination synergy scores 

# Motivation

The improved therapeutic outcome and reduced adverse effects of synergistic drug combinations have turned them into standard of care in many cancer types. Given the wealth of relevant information provided by high-throughput screening studies, the costly experimental design of these combinations can now be guided by advanced computational tools. In this context, we present MARSY, a deep-multitask learning method that predicts the level of synergism between drug pairs tested on cancer cell lines. Using gene expression to characterize cancer cell lines and induced signature profiles to represent drug pairs, MARSY learns a distinct set of embeddings to obtain multiple views of the features. Precisely, a representation of the entire combination and a representation of the drug pair are learned in parallel. These representations are then fed to a multitask network that predicts the synergy score of the drug combination alongside single drug responses. A thorough evaluation of MARSY revealed its superior performance compared to various state-of-the-art and traditional computational methods. A detailed analysis of the design choices of our framework demonstrated the predictive contribution of the learned embeddings by this model.

# Requirements

In order to run the MARSY.py code, Python 3.8 need to be installed. In addition, the code uses the following python modules/libraries which also need to be installed:

- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)

# Data Files

## Description of training and test sets

The input vector of each sample in the provided training and testing sets is the concatenation of both drugs' features (signature) and cancer cell line features (gene expression). The drug signature is represented by the first 3912 features (1956 features per drug) of the vector while the gene expression of the cell line is represented by the next 4639 features. To make MARSY as invariant as possible to the order of the drug features, each sample is provided twice with the order of each drug reversed. A visual explanation of the elements composing each sample is shown in the input sample [figure](Input_Sample.pdf).

Example of input datasets:

|  | 1956 drug signature features | 1956 drug signature features | 4639 gene expression features |
| --- | --- | --- | --- |
| Sample_1 | Drug 1| Drug 2 | Cell Line A |
| Sample_2 | Drug 2 | Drug 1|  Cell Line A  |
| Sample_3 | Drug 3| Drug 4 | Cell Line B  |
| Sample_4 | Drug 4| Drug 3 | Cell Line B  |

The MARSY [implementation](MARSY.py) contains a data prepartion function that converts the input samples into the required format used as input to the model. This format is designed such that MARSY can learn different embeddings from the input features using its two encoders as shown in the architecture [figure](Architecture_MARSY.pdf)

## Description of targets

MARSY is a multitask learning based model. Thus, the learning process is based not only on the synergy score of each drug combination but also on the response of each drug. Therfore, the target files are composed of three columns such that the first column represents the synergy scores (ZIP or any other score) while the second and third columns represent the the single responses of drug 1 and drug 2 respectively.

Example of targets for each sample:

|  | Synergy Score | Single Response of Drug 1 | Single Response of Drug 2 |
| --- | --- | --- | --- |
| Sample_1 | 2.80| -4.54 | 0.09 |
| Sample_2 | 2.80 | 0.09 |  -4.54 |
| Sample_3 | 6.18| 29.12 | -30.99 |
| Sample_4 | 6.18 | -30.99 |  29.12 |

# Running MARSY

The MARSY [file](MARSY.py) provided contains a basic implementation of MARSY. The code first reads the provided data files and then, uses a data preparation function that converts the input data sets into the appropriate format. It is also composed of the implementation of the deep multitask neural network. The parameters of this model are set in the code following our design choices. Finally, an implementation of the training and a prediction example are also added.

To run the code, the script along with all the data files in this repository need to be in the same folder. The parameters of the model can be changed directly in the script.
