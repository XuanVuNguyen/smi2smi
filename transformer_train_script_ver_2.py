from smi2smi.data import TokenLookup, file_to_chrs, pad_sequence
from smi2smi.transformer_ver_2 import build_model
from smi2smi.train import fit_and_save
import os
import json

model_config = {
    'encoder_depth':3,
    'decoder_depth':3,
    'vocab_size':54,
    'dropout':0.1,
    'embedding_share':True,
    
    'conv_config':{
        'depth':3,
        'kernel_sizes':[1, 3, 5],
        'n_filters':64
    },
    'block_config':{
        'model_dim':64,
        'num_heads':8,
        'key_dim':64,
        'value_dim':64,
        'ff_inner_dim':512
    },
    'softmax_temperature':1.3,
}

data = {
    'data_dir':'data_wo_reaction_label/',
    'max_input_length':200,
    'max_output_length':200
}

training_config = {
    'batch_size':64,
    'epochs':1000,
    'optimizer':{
        'class_name':'Adam',
        'config':{'learning_rate':1e-04,
                  'beta_1':0.9,
                  'beta_2':0.98,
                  'epsilon':1e-09,
                  'clipnorm':5.0
                 }
    },

    'exponential_decay_lr':False,
    'exponential_decay_lr_config':{
        'initial_learning_rate':0.0001,
        'decay_rate': 0.99,
        'decay_steps':100,
        'min_learning_rate':1e-12,
        'staircase':False
    },

    'cyclic_lr':False,
    'cyclic_lr_config':{
        'base_lr':0., #overwrite optimzier.lr
        'max_lr':0.001,
        'epochs_per_cycle':10,
        'warm_up_epochs':3
    },
    
    'transformer_lr':True,
    'transformer_lr_config':{
        'warm_up_epochs':26,
        'model_dim':64
    },

    'save_freq':4, 
    # number of epochs to save and evaluate, should be equal'epochs_per_cycle' if 'cyclic_lr' is True
    'save_dir':'model_checkpoints/Sep7_16h03_3_CNN/',
    'initial_epoch':0}

token_lookup = TokenLookup(os.path.join(data['data_dir'], 'vocab'))

train_source_chrs = file_to_chrs(os.path.join(data['data_dir'], 'train_sources'))
train_target_chrs = file_to_chrs(os.path.join(data['data_dir'], 'train_targets'))
valid_source_chrs = file_to_chrs(os.path.join(data['data_dir'], 'valid_sources'))
valid_target_chrs = file_to_chrs(os.path.join(data['data_dir'], 'valid_targets'))

train_encoder_inputs = pad_sequence(token_lookup.chrs_to_ids(train_source_chrs), 
                                    data['max_input_length'])
train_decoder_inputs = pad_sequence(
    token_lookup.chrs_to_ids(token_lookup.sos_prefix(train_target_chrs)),
    data['max_output_length'])
train_decoder_outputs = pad_sequence(
    token_lookup.chrs_to_ids(token_lookup.eos_suffix(train_target_chrs)),
    data['max_output_length'])

valid_encoder_inputs = pad_sequence(token_lookup.chrs_to_ids(valid_source_chrs), 
                                    data['max_input_length'])
valid_decoder_inputs = pad_sequence(
    token_lookup.chrs_to_ids(token_lookup.sos_prefix(valid_target_chrs)),
    data['max_output_length'])
valid_decoder_outputs = pad_sequence(
    token_lookup.chrs_to_ids(token_lookup.eos_suffix(valid_target_chrs)),
    data['max_output_length'])

if training_config['initial_epoch']==0:
    os.mkdir(training_config['save_dir'])
    with open(os.path.join(training_config['save_dir'], 'config.txt'), 'w') as params_file:
        json.dump({'model_config':model_config, 'data':data, 'training_config':training_config}, params_file, indent=4)
else:
    with open(os.path.join(training_config['save_dir'], 'config.txt'), 'r') as params_file:
        model_config = json.load(params_file)['model_config']
        
model = build_model(model_config)

fit_and_save(
    model=model,
    train_params=training_config,
    input_seqs=[train_encoder_inputs, train_decoder_inputs],
    output_seq=train_decoder_outputs,
    valid_input_seqs=[valid_encoder_inputs, valid_decoder_inputs],
    valid_output_seq=valid_decoder_outputs)