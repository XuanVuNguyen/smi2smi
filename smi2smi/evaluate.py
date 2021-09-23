import numpy as np
import tensorflow as tf
import os
from rdkit.Chem import AllChem

def evaluate(ground_truths, candidate_list, canonicalize=True, k=20):
    '''
    Args:
      ground_truth: shape=(n_samples)
      predicted_seqs: list(shape=(beam_width, )    
    '''
    def _canonicalize(smi):
        return AllChem.MolToSmiles(AllChem.MolFromSmiles(smi))
    def _evaluate_single_prediction(ground_truth, candidates, canonicalize=True):
        '''
        Args:
          ground_truth: string
          candidates: array(top_k)
        '''
        correct_ids, = np.where(candidates==ground_truth)
        em_score = int(correct_ids.size > 0)
        return correct_ids, em_score
    def _grammar_check(smi):
        if AllChem.MolFromSmiles(smi) is None:
            return False
        else:
            return True
    def _invalid_smi_filter_and_canonicalize(top_k_candidate, canonicalize):
        invalid_smis = []
        valid_smis = []
        maybe_canon_valid_smis = []
        for i, smi in enumerate(top_k_candidate):
            if not _grammar_check(smi):
                invalid_smis.append(smi)
            elif _grammar_check(smi):
                valid_smis.append(smi)
                if canonicalize:
                    canonicalized_smi = _canonicalize(smi)
                    maybe_canon_valid_smis.append(canonicalized_smi)
                elif not canonicalize:
                    maybe_canon_valid_smis.append(smi)
        return np.array(valid_smis), np.array(maybe_canon_valid_smis), np.array(invalid_smis)
    
    n_samples = ground_truths.shape[0]
    n_candidates = 0
    total_em_score = 0
    n_potential = 0
    n_valid_smis = 0
    n_invalid_smis = 0
    correct_predictions = []
    all_correct_prediction_ids = []
    potential_candidate_list = []
    invalid_smi_list = []
    index = []
    for i in range(n_samples):
        top_k_candidate = candidate_list[i][:k]
        if canonicalize:    
            ground_truth = _canonicalize(ground_truths[i])
        elif not canonicalize:
            ground_truth = ground_truths[i]
        valid_smis, maybe_canon_valid_smis, invalid_smis = \
            _invalid_smi_filter_and_canonicalize(top_k_candidate, canonicalize)
        correct_prediction_ids, em_score = \
            _evaluate_single_prediction(ground_truth, maybe_canon_valid_smis)
        
        potential_candidates = np.delete(valid_smis, correct_prediction_ids)
        
        correct_predictions.append(valid_smis[correct_prediction_ids])
        all_correct_prediction_ids.append(correct_prediction_ids)
        potential_candidate_list.append(potential_candidates)
        invalid_smi_list.append(invalid_smis)
        n_candidates += top_k_candidate.shape[0] 
        total_em_score += em_score
        n_potential += potential_candidates.shape[0]
        n_valid_smis += valid_smis.shape[0]
        n_invalid_smis += invalid_smis.shape[0]
        
        index.append(i)    
    accuracy = total_em_score / n_samples
    return {'top_k':k,
            'n_samples':n_samples,
            'n_candidates':n_candidates,
            'total_em_score':total_em_score,
            'accuracy':accuracy,
            'n_valid_smis':n_valid_smis,
            'n_invalid_smis':n_invalid_smis,
            'correct_predictions':np.array(correct_predictions, dtype=object), # list
            'correct_prediction_ids':np.array(all_correct_prediction_ids, dtype=object),
            'potential_candidates':np.array(potential_candidate_list, dtype=object), # list of arrays
            'invalid_candidates':np.array(invalid_smi_list, dtype=object),
            'index':np.array(index)
            } 

def rxn_sort(results, rxn_labels):
    all_rxn_results = [results]
    for i in range(10):
        rxn_class_ids, = np.where(rxn_labels==f'<RX_{i+1}>'.encode('ascii'))
        rxn_results = {}
        rxn_results['top_k'] = results['top_k']
        rxn_results['n_samples'] = rxn_class_ids.size
        rxn_results['correct_predictions'] = results['correct_predictions'][rxn_class_ids]
        rxn_results['correct_prediction_ids'] = results['correct_prediction_ids'][rxn_class_ids]
        rxn_results['potential_candidates'] = results['potential_candidates'][rxn_class_ids]
        rxn_results['invalid_candidates'] = results['invalid_candidates'][rxn_class_ids]
        rxn_results['total_em_score'] = sum(array.size>0 for array in rxn_results['correct_predictions'])
        rxn_results['accuracy'] = rxn_results['total_em_score']/rxn_results['n_samples']
        rxn_results['n_valid_smis'] = \
            sum(rxn_results['correct_predictions'][i].size+rxn_results['potential_candidates'][i].size\
                for i in range(len(rxn_class_ids)))
        rxn_results['n_invalid_smis'] = sum(array.size for array in rxn_results['invalid_candidates'])
        rxn_results['n_candidates'] = rxn_results['n_valid_smis']+rxn_results['n_invalid_smis']
        rxn_results['index'] = rxn_class_ids
        all_rxn_results.append(rxn_results)
    return all_rxn_results

def print_rxn_class_results(results, ground_truths, source_smis, rxn_label, result_dir):
    with open(os.path.join(result_dir, rxn_label+'.txt'), 'w') as file:
        def _write_nested_array(nested_array, index_in_sub_list=None):
            for i, prediction_array in enumerate(nested_array):
                file.write('{:04d}.\n'.format(results['index'][i]))
                for j, prediction in enumerate(prediction_array):
                    if index_in_sub_list is not None:
                        rank = ' (rank_{:02d})'.format(index_in_sub_list[i][j]+1)
                    else:
                        rank = ''
                    file.write(prediction+'>>'+source_smis[results['index'][i]].decode('ascii')+rank+'\n')
                file.write('Ground truth:\n'+ground_truths[results['index'][i]].decode('ascii')\
                           +'>>'+source_smis[results['index'][i]].decode('ascii')+'\n')
            
        file.write('====={}=====\n'.format(rxn_label.upper()))
        file.write('\nContents: METRICS, CORRECT PREDICTIONS, POTENTIAL PREDICTIONS, INVALID SMILES.\n')
        file.write('\nMETRICS:\n')
        file.write('Number of samples: {}\n'.format(results['n_samples']))
        file.write('Number of candidates: {}\n'.format(results['n_candidates']))
        file.write('Number of exact matches: {}\n'.format(results['total_em_score']))
        file.write('Accuracy: {:.06f}\n'.format(results['accuracy']))
        file.write('Number of valid SMILES: {}\n'.format(results['n_valid_smis']))
        file.write('Number of invalid SMILES: {}\n\n'.format(results['n_invalid_smis']))
        
        file.write('CORRECT PREDICTIONS:\n')
        _write_nested_array(results['correct_predictions'], results['correct_prediction_ids'])
        
        file.write('\nPOTENTIAL PREDICTIONS:\n')
        _write_nested_array(results['potential_candidates'])
        
        file.write('\nINVALID SMILES:\n')
        _write_nested_array(results['invalid_candidates'])
            
def print_results(all_results, ground_truths, source_smis, beam_results_dir):
    results_dir = os.path.join(beam_results_dir, 'top_{}_results/'.format(all_results[0]['top_k']))
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    label = ['general']+[f'rx_{i+1}' for i in range(10)]
    for i in range(11):
        print_rxn_class_results(all_results[i], ground_truths, source_smis, label[i], results_dir)