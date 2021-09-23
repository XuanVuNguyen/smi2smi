from smi2smi.data_reader import MolecularTokenizer
from smi2smi.recurrent import build_model
from smi2smi.train import fit_and_save
import os
import json
import pickle

model_config = {
    'vocab_size':3100,
    'embedding_dim':512,
    'embedding_share':True,
    'source_reverse':False,
    'cnn_config':{
        'depth':1,
        'kernel_sizes':[1, 3, 5],
        'n_filters':512,
        'residual_connect':False,
        'layer_norm':False
    },
    'encoder_rnn_config':{
        'hidden_dim':512,
        'depth':1,
        'dropout_rate_input':0.2,
        'dropout_rate_output':0.0,
        'residual_connect':None,
        'bidirectional_encoding':True
    },
    
    'state_bridge':'zero',
    
    'decoder_rnn_config':{
        'hidden_dim':512,
        'depth':1,
        'dropout_rate_input':0.2,
        'dropout_rate_output':0.0,
        'residual_connect':None
    },
    'attention_config':{
        'attention_mechanism':'Luong',
        'attention_dim':512,
        'append_context':True
    }
}

data = {
    'language':'smiles',
    'tokenizer':'bpe',
    'bpe_file':'SPE_ChEMBL.txt',
    'include_reaction_label':True,
    'frequency_threshold':3,
    'data_dir':'USPTO-50K-SMILES/',
    'max_input_length':200,
    'max_output_length':200
}

training_config = {
    'batch_size':32,
    'epochs':1000,
    'optimizer':{
        'class_name':'Adam',
        'config':{'learning_rate':1e-03,
                  'beta_1':0.9,
                  'beta_2':0.999,
                  'epsilon':1e-07,
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
    
    'transformer_lr':False,
    'transformer_lr_config':{
        'warm_up_epochs':26,
        'model_dim':64
    },

    'save_freq':4, 
    # number of epochs to save and evaluate, should be equal'epochs_per_cycle' if 'cyclic_lr' is True
    'save_dir':'model_checkpoints/Sep22_22h31_rnn_smiles_bpe_512/',
    'initial_epoch':0}

if training_config['initial_epoch']==0:
    os.mkdir(training_config['save_dir'])
    with open(os.path.join(training_config['save_dir'], 'config.txt'), 'w') as params_file:
        json.dump({'model_config':model_config, 'data':data, 'training_config':training_config}, params_file, indent=4)
else:
    with open(os.path.join(training_config['save_dir'], 'config.txt'), 'r') as params_file:
        model_config = json.load(params_file)['model_config']

tokenizer = MolecularTokenizer(data['language'], 
                               data['tokenizer'],
                               data['bpe_file'])

train_and_valid_files = ['train_sources', 'valid_sources', 'train_targets', 'valid_targets']
train_sources, valid_sources, train_targets, valid_targets = \
    [tokenizer.tokenize_from_file(os.path.join(data['data_dir'], file), 
                                  data['include_reaction_label']) 
     for file in train_and_valid_files]

tokenizer.fit_on_texts(train_sources + valid_sources + train_targets + valid_targets,
                       data['frequency_threshold'])

with open(os.path.join(training_config['save_dir'], 'tokenizer.pkl'), 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file, pickle.HIGHEST_PROTOCOL)

train_encoder_inputs, valid_encoder_inputs = [tokenizer.chars_to_indices(sources,
                                                                         data['max_input_length']) 
                                              for sources in [train_sources, valid_sources]]

train_decoder_inputs, valid_decoder_inputs = [tokenizer.chars_to_indices(tokenizer.sos_prepend(targets),
                                                                         data['max_output_length']) 
                                              for targets in [train_targets, valid_targets]]

train_decoder_outputs, valid_decoder_outputs = [tokenizer.chars_to_indices(tokenizer.eos_append(targets),
                                                                           data['max_output_length']) 
                                              for targets in [train_targets, valid_targets]]


        
model = build_model(model_config)

fit_and_save(
    model=model,
    train_params=training_config,
    input_seqs=[train_encoder_inputs, train_decoder_inputs],
    output_seq=train_decoder_outputs,
    valid_input_seqs=[valid_encoder_inputs, valid_decoder_inputs],
    valid_output_seq=valid_decoder_outputs)