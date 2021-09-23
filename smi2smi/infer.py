#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import os
import time

# def save_beam_search_results(final_output, 
#                              beam_results,
#                              infer_params):
#     save_dir = infer_params['result_dir']\
#         +'_ck-{:05d}'.format(infer_params['at_epoch'])\
#         +'_beam_{}'.format(infer_params['beam_width'])
#     os.mkdir(save_dir)
#     with open(os.path.join(save_dir, 'infer_params.txt'), 'w') as params_file:
#         json.dump({'infer_params':infer_params}, params_file, indent=4)
#     save_path = os.path.join(save_dir, 'predictions')
#     np.savez(save_path, 
#              final_output=final_output,
#              scores=beam_results['scores'],
#              beam_predicted_ids=beam_results['predicted_ids'],
#              beam_parents_ids=beam_results['parent_ids'])

def infer_and_save(inference_model, indices_to_strings, padded_inputs, batch_size, save_path):
    def _indices_to_strings_for_beam_outputs(beam_outputs):
        strings = indices_to_strings(np.reshape(beam_outputs, 
                                                (beam_outputs.shape[0]*beam_outputs.shape[1], -1)))
        return np.reshape(strings, (beam_outputs.shape[0], beam_outputs.shape[1]))
                          
    os.mkdir(os.path.dirname(save_path))
    start_time = time.time()
    inputs_dataset = tf.data.Dataset.from_tensor_slices(padded_inputs)
    all_results = []
    for inputs in iter(inputs_dataset.batch(1)):
        results = inference_model(inputs)
    #     results includes ids, lengths, log_probs, attention, state
        all_results.append(results)
    inference_time = time.time() - start_time
    print('Inference time: {:.04f}s'.format(inference_time))
    

    padded_ids = [np.pad(i.ids,
                         [[0, 0], 
                          [0, 0], 
                          [0, inference_model.configs['infer_config']['max_decode_length']-i.ids.shape[2]]],
                         constant_values=2) for i in all_results]
    
    output_ids = np.concatenate(padded_ids, axis=0) # shape==(N, beamwidth, max_len)
    
    filtered_output_ids = filter_output_ids(output_ids)
    
    output_strings = _indices_to_strings_for_beam_outputs(filtered_output_ids)
    
    lengths = np.concatenate([i.lengths for i in all_results], axis=0)
    log_probs = np.concatenate([i.log_probs for i in all_results], axis=0)
    
    if all_results[0].attention is not None:
        attention = np.concatenate([i.attention for i in all_results], axis=0)
        np.savez(save_path,
                 ids=output_ids,
                 strings=output_strings,
                 lengths=lengths,
                 attention=attention,
                 log_probs=log_probs,
                 inference_time=np.array(inference_time))
        print('Results file\'s keys: ids, strings, lengths, attention, log_probs, inference_time.')
    else:
        np.savez(save_path,
                 ids=output_ids,
                 strings=output_strings,
                 lengths=lengths,
                 log_probs=log_probs,
                 inference_time=np.array(inference_time))
        print('Results file\'s keys: ids, strings, lengths, log_probs, inference_time.')
    
def filter_output_ids(results_ids):
    '''
    Args:
     results_ids: np.ndarray shape==(N, beamwidth, len).
    Returns:
     np.ndarray shape==(N, beamwidth, len).'''
    incomplete_mask = results_ids[:, :, -1]==2
    zero_contain_mask = np.logical_not(np.any(results_ids==0, axis=-1))
    masked_ids = results_ids*incomplete_mask[:, :, np.newaxis]
    masked_ids = masked_ids*zero_contain_mask[:, :, np.newaxis]
    return masked_ids*(masked_ids!=2)