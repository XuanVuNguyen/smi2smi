# Overview
This project is my internship project for my bachelor graduation. You read see my thesis [here](https://drive.google.com/file/d/1TrVG1kOC_gD-_nRe7x4Mb1rwvsEXr2be/view?usp=sharing).

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
RNN and Transformer share a same [infer_script.py](./infer_script.py) file. Open the file and tune the hyper-parameters to your preference, save and run:

```
python infer_script.py
```

Inferring results are saved as a `.npz` file under the [predictions_and_evaluation](./predictions_and_evaluation) directory.
The content of the `.npz` file includes the following key words:

* `'ids'` The inferred sequences representing the reactants in the form of character indices.
* `'strings'` The inferred sequences representing the reactants in the form of characters.
* `'lengths'` Lengths of the inferred sequences.
* `'log_probs'` Log probability of the every characters of the output sequences.
* `'inference_time'` Inference time in second.
* `'attention'` The attention matrices. This key word will not be included if no attention mechanism is used.

## Evaluate
Evaluation is done by examining the `.npz` file from inference step. Open [evaluate_script.py](./evaluate_script.py), tune the arguments and run:

```
python evaluate_script.py
```

The evaluation results are saved in a `.csv` within the same directory as the specified `.npz` file.