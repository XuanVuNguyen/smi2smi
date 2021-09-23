#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, IntegerLookup
import numpy as np


# In[2]:


def file_to_chrs(data_file_path):
    with open(data_file_path, mode='r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    tokens = tf.strings.split(lines)
    return tokens


# In[3]:


def target_file_to_chrs(target_data_path):
    prepend_token = 'SEQUENCE_START'
    append_token = 'SEQUENCE_END'
    seq_tokens = file_to_chrs(target_data_path)
    sandwiched_seq_tokens = tf.concat([tf.fill([seq_tokens.shape[0], 1], prepend_token), 
                                    seq_tokens, 
                                    tf.fill([seq_tokens.shape[0], 1], append_token)], 
                                    axis=-1)
    target_seq_input_tokens = sandwiched_seq_tokens[:, :-1]
    target_seq_output_tokens = sandwiched_seq_tokens[:, 1:]
    return target_seq_input_tokens, target_seq_output_tokens


# In[4]:


class TokenLookup:
    def __init__(self, vocab_file_path):
        with open(vocab_file_path, mode='r') as vocab_file:
            self.vocab_og = vocab_file.read().splitlines()
            
        self._special_chrs = ['SEQUENCE_START', 'SEQUENCE_END']
        self._vocab_full = self._special_chrs + self.vocab_og
        self._vocab_size = len(self._vocab_full)
        self._chrs_to_ids = StringLookup(vocabulary=self._vocab_full,
                                        oov_token='UNK',
                                        num_oov_indices=0)
        self._ids_to_chrs = StringLookup(
            vocabulary=self._vocab_full,
            oov_token='UNK',
            num_oov_indices=0,
            invert=True)
    
    def sos_prefix(self, chrs):
        return tf.concat([tf.fill([chrs.shape[0], 1], self.special_chrs[0]), 
                            chrs],
                         axis=-1)
                         
    def eos_suffix(self, chrs):
        return tf.concat([chrs,
                          tf.fill([chrs.shape[0], 1], self.special_chrs[1])],
                         axis=-1)
        
    def chrs_to_ids(self, tokens):
        return tf.cast(self._chrs_to_ids(tokens), dtype=tf.int32)
    def ids_to_chrs(self, ids):
        return self._ids_to_chrs(ids)
    @property
    def vocab_full(self):
        return self._vocab_full
    @property
    def vocab_size(self):
        return self._vocab_size
    @property
    def special_chrs(self):
        return self._special_chrs

def pad_sequence(tokens, max_length):
    return tokens.to_tensor(shape=(tokens.shape[0], max_length))

def load_beam_results_file(file_dir):
    with np.load(file_dir+'predictions.npz', allow_pickle=True) as beam_output:
        final_output = beam_output['final_output']
        scores = beam_output['scores']
        beam_predicted_ids = beam_output['beam_predicted_ids']
        beam_parents_ids = beam_output['beam_parents_ids']
        return {'final_output':final_output,
                'scores':scores,
                'beam_predicted_ids':beam_predicted_ids,
                'beam_parents_ids':beam_parents_ids}
    
def read_beams(final_output, lookup):
    def _incompleted_candidate_filter(cand_ints, end_token_id):
        # even if the entire beams is eliminated, it remains as an empty array.
        completed_cand = tf.boolean_mask(cand_ints, cand_ints[:, -1]==end_token_id)
        return completed_cand
    def _0_containing_filter(cand_ints):
        mask = tf.math.logical_not(tf.math.reduce_any(cand_ints==0, -1))
        filtered_cand = tf.boolean_mask(cand_ints, mask)
        return filtered_cand
    end_token_id = lookup.str_to_ids(lookup.special_tokens[1])
    candidates = []
    for output in final_output:
        cand_ints = tf.transpose(output, [1, 0])
        filtered_cand_int = _incompleted_candidate_filter(cand_ints, end_token_id)
        filtered_cand_int = _0_containing_filter(filtered_cand_int)
        unpadded_cand = tf.RaggedTensor.from_tensor(filtered_cand_int, ragged_rank=1, padding=end_token_id)
        candidate = tf.strings.reduce_join(lookup.ids_to_str(unpadded_cand), 1)
        candidates.append(np.char.decode(candidate.numpy().astype('S'), 'ascii'))
    return candidates

def get_rxn_class(test_source_tokens):
    source_smis = tf.strings.reduce_join(test_source_tokens[:, 1:], 1).numpy()
    rxn_labels = test_source_tokens.to_tensor()[:, 0].numpy()
    return source_smis, rxn_labels



