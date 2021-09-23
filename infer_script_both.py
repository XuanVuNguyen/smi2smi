from smi2smi.data_reader import MolecularTokenizer
from smi2smi.transformer_ver_2 import InferenceModel as InferringTransformer
from smi2smi.recurrent import InferenceModel as InferringRecurrent
from smi2smi.infer import infer_and_save
import numpy as np
import os
import json
import pickle

MODEL_TYPE = 'recurrent'

inference_config = {
    'beam_width':10,
    'length_penalty':0.2,
    'max_decode_length':140,
    'model_dir':'model_checkpoints/Sep17_18h19_rnn_smiles_characterwise/',
    'epoch_number':148
}

data = {
    'test_sources':'USPTO-50K-SMILES/test_sources',
    'batch_size':64,
    'results_dir':'predictions_and_evaluation/USPTO-50K'
}


# Build inference model
with open(os.path.join(inference_config['model_dir'], 'config.txt'), 'r') as config_file:
    config = json.load(config_file)
    model_config = config['model_config']
    train_data_config = config['data']

if MODEL_TYPE == 'recurrent':
    inference_model = InferringRecurrent(model_config, inference_config)
elif MODEL_TYPE == 'transformer':
    inference_model = InferringTransformer(model_config, inference_config)
    
inference_model.build_and_load_weights()

with open(os.path.join(inference_config['model_dir'], 'tokenizer.pkl'), 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Preprocess inputs

test_source_chrs = tokenizer.tokenize_from_file(data['test_sources'],
                                                train_data_config['include_reaction_label'])

test_inputs = tokenizer.chars_to_indices(test_source_chrs,
                                         train_data_config['max_input_length'])

# Path to save results
results_sub_dir_prefix = os.path.basename(inference_config['model_dir'][:-1]) \
    if inference_config['model_dir'][-1]=='/' \
    else os.path.basename(inference_config['model_dir'])

results_sub_dir = results_sub_dir_prefix + '_epoch-{:05d}_bw-{}_lenpen-{}_maxdeclen-{}'.format(
    inference_config['epoch_number'],
    inference_config['beam_width'],
    inference_config['length_penalty'],
    inference_config['max_decode_length'])

results_path = os.path.join(data['results_dir'], results_sub_dir, 'prediction')

# Infer and save
infer_and_save(inference_model, 
               tokenizer.indices_to_strings,
               test_inputs, 
               data['batch_size'], 
               results_path)