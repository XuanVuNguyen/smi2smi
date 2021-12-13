import os
import csv
from collections import defaultdict
import numpy as np
import tensorflow as tf
from rdkit.Chem import AllChem

from smi2smi.data_reader import canonicalize, grammar_check

# def grammar_check_for_beam_output(predicted_strings):
#     for strg_beam in predicted_strings:
        

def evaluate(predicted_strings, groundtruths, top_k, selfies):
    '''
    
    ''' 
    n_examples = groundtruths.shape[0]
    canonicalized_predicted_strings = canonicalize(predicted_strings, selfies=selfies)
    n_exact_matches = np.sum(np.any(canonicalized_predicted_strings[:, :top_k]==groundtruths[:, np.newaxis],
                                  axis=-1))
                           
    accuracy = n_exact_matches/n_examples
    
    n_candidates = np.sum(predicted_strings!='')
    n_valid_sequences = np.sum(canonicalized_predicted_strings!='')
    n_invalid_sequences = n_candidates - n_valid_sequences
    return {'n_examples':n_examples,
            'n_candidates':n_candidates,
            'n_exact_matches':n_exact_matches,
            'accuracy':accuracy,
            'n_valid_sequences':n_valid_sequences,
            'n_invalid_sequences':n_invalid_sequences}

def rxn_sort(sequences, source_rxn_labels: list):
    available_labels = [f'<RX_{i+1}>' for i in range(10)]
    source_rxn_labels = np.array(source_rxn_labels)
    sorted_sequences = {}
    for available_label in available_labels:
        idx, = np.where(source_rxn_labels==available_label)
        sampled_sequences = sequences[idx]
        sorted_sequences[available_label] = sampled_sequences
    return sorted_sequences

def rxn_aware_evaluate(predicted_sequences:np.ndarray, 
                       groundtruths:np.ndarray, 
                       top_k, 
                       source_rxn_labels, 
                       selfies):
    outputs = []
    overall_eval = evaluate(predicted_sequences, groundtruths, top_k, selfies)
    overall_eval['reaction_class'] = 'overall'
    outputs.append(overall_eval)
    sorted_groundtruths = rxn_sort(groundtruths, source_rxn_labels)
    sorted_predicted_sequences = rxn_sort(predicted_sequences, source_rxn_labels)
    for rxn_class, rxn_sequence_set in sorted_predicted_sequences.items():
        rxn_eval = evaluate(rxn_sequence_set, sorted_groundtruths[rxn_class], top_k, selfies)
        rxn_eval['reaction_class'] = rxn_class
        outputs.append(rxn_eval)
    return outputs        

def writeout_evaluation_results(eval_results, write_dir):
    fieldnames = ['reaction_class', 'n_examples', 'n_candidates', 'n_exact_matches', 'accuracy',
                  'n_valid_sequences', 'n_invalid_sequences']
    with open(write_dir, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_results)
    print(f'Evaluate results are saved at {os.path.abspath(write_dir)}')
    

# def rxn_sort(results, rxn_labels):
#     all_rxn_results = [results]
#     for i in range(10):
#         rxn_class_ids, = np.where(rxn_labels==f'<RX_{i+1}>'.encode('ascii'))
#         rxn_results = {}
#         rxn_results['top_k'] = results['top_k']
#         rxn_results['n_samples'] = rxn_class_ids.size
#         rxn_results['correct_predictions'] = results['correct_predictions'][rxn_class_ids]
#         rxn_results['correct_prediction_ids'] = results['correct_prediction_ids'][rxn_class_ids]
#         rxn_results['potential_candidates'] = results['potential_candidates'][rxn_class_ids]
#         rxn_results['invalid_candidates'] = results['invalid_candidates'][rxn_class_ids]
#         rxn_results['total_em_score'] = sum(array.size>0 for array in rxn_results['correct_predictions'])
#         rxn_results['accuracy'] = rxn_results['total_em_score']/rxn_results['n_samples']
#         rxn_results['n_valid_smis'] = \
#             sum(rxn_results['correct_predictions'][i].size+rxn_results['potential_candidates'][i].size\
#                 for i in range(len(rxn_class_ids)))
#         rxn_results['n_invalid_smis'] = sum(array.size for array in rxn_results['invalid_candidates'])
#         rxn_results['n_candidates'] = rxn_results['n_valid_smis']+rxn_results['n_invalid_smis']
#         rxn_results['index'] = rxn_class_ids
#         all_rxn_results.append(rxn_results)
#     return all_rxn_results

# def print_rxn_class_results(results, ground_truths, source_smis, rxn_label, result_dir):
#     with open(os.path.join(result_dir, rxn_label+'.txt'), 'w') as file:
#         def _write_nested_array(nested_array, index_in_sub_list=None):
#             for i, prediction_array in enumerate(nested_array):
#                 file.write('{:04d}.\n'.format(results['index'][i]))
#                 for j, prediction in enumerate(prediction_array):
#                     if index_in_sub_list is not None:
#                         rank = ' (rank_{:02d})'.format(index_in_sub_list[i][j]+1)
#                     else:
#                         rank = ''
#                     file.write(prediction+'>>'+source_smis[results['index'][i]].decode('ascii')+rank+'\n')
#                 file.write('Ground truth:\n'+ground_truths[results['index'][i]].decode('ascii')\
#                            +'>>'+source_smis[results['index'][i]].decode('ascii')+'\n')
            
#         file.write('====={}=====\n'.format(rxn_label.upper()))
#         file.write('\nContents: METRICS, CORRECT PREDICTIONS, POTENTIAL PREDICTIONS, INVALID SMILES.\n')
#         file.write('\nMETRICS:\n')
#         file.write('Number of samples: {}\n'.format(results['n_samples']))
#         file.write('Number of candidates: {}\n'.format(results['n_candidates']))
#         file.write('Number of exact matches: {}\n'.format(results['total_em_score']))
#         file.write('Accuracy: {:.06f}\n'.format(results['accuracy']))
#         file.write('Number of valid SMILES: {}\n'.format(results['n_valid_smis']))
#         file.write('Number of invalid SMILES: {}\n\n'.format(results['n_invalid_smis']))
        
#         file.write('CORRECT PREDICTIONS:\n')
#         _write_nested_array(results['correct_predictions'], results['correct_prediction_ids'])
        
#         file.write('\nPOTENTIAL PREDICTIONS:\n')
#         _write_nested_array(results['potential_candidates'])
        
#         file.write('\nINVALID SMILES:\n')
#         _write_nested_array(results['invalid_candidates'])
            
# def print_results(all_results, ground_truths, source_smis, beam_results_dir):
#     results_dir = os.path.join(beam_results_dir, 'top_{}_results/'.format(all_results[0]['top_k']))
#     if not os.path.exists(results_dir):
#         os.mkdir(results_dir)
#     label = ['general']+[f'rx_{i+1}' for i in range(10)]
#     for i in range(11):
#         print_rxn_class_results(all_results[i], ground_truths, source_smis, label[i], results_dir)