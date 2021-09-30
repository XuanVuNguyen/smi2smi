# Overview
This project is my internship project for my bachelor graduation.

In this project, I aimed at using machine learning to solve the problem of retrosynthesis analysis, 
i.e. given a target organic molecule, a set of reactants compounds, which can react to form the target, are deduced.
This is one of the main task in Computer-Assisted Synthesis Planning.

In particular, the models take a SMILES or SELFIES representations of a target molecule as inputs, and return another SMILES or SELFIES representing 
the reactants as output. Since SMILES and SELFIES are text strings, Sequence-to-sequence formalism, such as Recurrent Neural Network (RNN) or Transformer, was 
implemented.

The models were trained on a open-sources database called USPTO-50K, which was preprocessed by Liu et al. (2017)

# Requirement
All the scripts are written in Python 3.7. The following packages are required:
- TensorFlow 2.5.0
- RDKit 2020.09.1
- OpenNMT-tf 2.20.1
- SELFIES 1.0.4
- SmilesPE 0.0.3

# Usage
## Training
To train a model, simply open the [rnn_train_script.py](./rnn_train_script.py) or [transformer_train_script.py](./transformer_train_script.py), corresponding with RNN 
and Transformer architectures respectively, tune the hyper-parameters, save and run:
```
python rnn_train_script.py
``` 
if RNN, or
```
python transformer_train_script.py
```
if Transformer.

Checkpoints during training are saved in the [model_checkpoints](./model_checkpoints) directory, along with the config of the models

## Inferring
RNN and Transformer share a same [infer_script.py](./infer_script.py) file. Open the file and tune the hyper-parameters to your preference, save and run.

Inferring results are saved as a .npz file under the [predictions_and_evaluation](./predictions_and_evaluation) directory.

## Evaluate

