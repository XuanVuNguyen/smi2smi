from smi2smi.data_old import TokenLookup, file_to_chrs, pad_sequence
from smi2smi.transformer_ver_2 import build_model, InferenceModel
from smi2smi.infer import infer_and_save
import numpy as np
import os
import json

inference_config = {
    'beam_width':10,
    'length_penalty':0.2,
    'max_decode_length':140,
    'max_encode_length':200,
    'model_dir':'model_checkpoints/Sep7_16h03_3_CNN/',
    'epoch_number':200
}

data = {
    'test_source_dir':'USPTO-50K_tokenized_wo_reaction_label/',
    'batch_size':64,
    'results_dir':'predictions_and_evaluation/USPTO-50K'
}



# Build inference model
with open(os.path.join(inference_config['model_dir'], 'config.txt'), 'r') as config_file:
    model_config = json.load(config_file)['model_config']
#     train_data = json.load(config_file)['data']
inference_model = InferenceModel(model_config, inference_config)
inference_model.build_and_load_weights()

# Preprocess inputs
token_lookup = TokenLookup(os.path.join(data['test_source_dir'], 'vocab'))
test_source_chrs = file_to_chrs(os.path.join(data['test_source_dir'], 'test_sources'))

test_inputs = pad_sequence(token_lookup.chrs_to_ids(test_source_chrs),
                           inference_config['max_encode_length'])

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
infer_and_save(inference_model, test_inputs, data['batch_size'], results_path)