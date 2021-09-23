import collections
import numpy as np
from tensorflow import keras
import selfies as sf
from SmilesPE.tokenizer import SPE_Tokenizer
from SmilesPE.pretokenizer import atomwise_tokenizer


class MolecularTokenizer:
    def __init__(self,
                 input_language,
                 tokenizer,
                 bpe_file=None):
        self._input_lang = input_language
        self._bpe_file = bpe_file
        
        if bpe_file is not None:
            with open(bpe_file, 'r') as file:
                self._bpe_tokenizer = SPE_Tokenizer(file)
                
        if self._input_lang == 'selfies':
            if tokenizer=='characterwise':
                raise ValueError('Character-wise tokenizer for SELFIES is non-sense!')
            if tokenizer=='bpe':    
                raise ValueError('BPE tokenizer for SELFIES will be supported in the future.')
            if tokenizer=='atomwise':
                self._tokenizer = self._selfies_atomwise_tokenizer

        if self._input_lang == 'smiles':
            if tokenizer=='characterwise':
                self._tokenizer = self._smiles_characterwise_tokenizer
            if tokenizer=='bpe':
                if self._bpe_file is not None:
                    with open(bpe_file, 'r') as file:
                        self._tokenizer = SPE_Tokenizer(file).tokenize
                else:
                    raise ValueError('A BPE file is required for BPE tokenizer.')
            
            if tokenizer=='atomwise':
                self._tokenizer = atomwise_tokenizer
    
    def _smiles_characterwise_tokenizer(self, string):
        return ' '.join(list(string))
    
    def _selfies_atomwise_tokenizer(self, string):
        return ' '.join(sf.split_selfies(string))
            
    def tokenize(self, strings):
        '''
        Args:
         strings: a list of strings.
        Returns:
         A list of tokenized strings with tokens seperated by blank character ' '.
        '''
        return [self._tokenizer(strg) for strg in strings]
    
    def fit_on_texts(self, 
                     tokenized_strings,
                     frequency_threshold):
        '''
        Args:
         strings: a list of strings.
         frequency_threshold: int. Tokens which appear less than this threshold are replaced by oov_token.
         include_reaction_label: bool, whether to include the ten reaction labels in the vocabulary.
        Returns:
         An instance of keras.preprocessing.text.Tokenizer.
        '''

        special_tokens=['[SEQ_START]', '[SEQ_END]']
        
        all_tokens = []
        for string in tokenized_strings:
            all_tokens += string.split(' ')
        token_count = collections.Counter(all_tokens).items()
        num_words = sum(count[1]>=frequency_threshold for count in token_count) \
                    + 2 + len(special_tokens)
        self._token_lookup = keras.preprocessing.text.Tokenizer(
            num_words=num_words,
            filters='',
            lower=False,
            oov_token='[UNKNOWN]')
        self._token_lookup.fit_on_texts(tokenized_strings)
        
        self._token_lookup.word_index = {k:v+len(special_tokens) for k, v in self._token_lookup.word_index.items()}
        self._token_lookup.word_index.update({k:v for k, v in zip(['']+special_tokens, 
                                                                  range(len(special_tokens)+1))})
        
        self._token_lookup.index_word = {v:k for k, v in self._token_lookup.word_index.items()}
        
        self._num_words = num_words
        
    def chars_to_indices(self, tokenized_strings, max_length=None):
        '''
        Args:
         strings: a list of strings.
         max_length: int. If None, pad to the length of the longest string.
        Returns:
         A np.array of indices, shape==(len(strings), max_length)
        '''
        sequences = self._token_lookup.texts_to_sequences(tokenized_strings)
        sequences = keras.preprocessing.sequence.pad_sequences(
            sequences=sequences,
            maxlen=max_length,
            dtype='int32',
            padding='post',
            truncating='post')
        
        return sequences
    
    def indices_to_chars(self, sequences):
        return np.array(self._token_lookup.sequences_to_texts(sequences))
    
    def indices_to_strings(self, sequences):
        return np.array([string.replace(' ', '') for string in self.indices_to_chars(sequences)])
    
    def tokenize_from_file(self, file_path, include_reaction_label=True):
        with open(file_path, 'r') as file:
            string_list = file.read().splitlines()
        if len(string_list[0].split(' '))==2:
            molecules = [string.split(' ')[-1] for string in string_list]
            if include_reaction_label:
                reaction_labels = [string.split(' ')[0] for string in string_list]
            else:
                reaction_labels = None
        elif len(string_list[0].split(' '))==1:
            reaction_labels = None
            molecules = string_list
            
        tokenized_strings = self.tokenize(molecules)
        if reaction_labels is not None:
            tokenized_strings = [' '.join([label, molecule]) \
                                 for label, molecule in zip(reaction_labels, tokenized_strings)]
        return tokenized_strings
    
    def sos_prepend(self, tokenized_strings):
        return [' '.join(['[SEQ_START]', string]) for string in tokenized_strings]
        
    def eos_append(self, tokenized_strings):
        return [' '.join([string, '[SEQ_END]']) for string in tokenized_strings]
        
    @property
    def token_lookup(self):
        return self._token_lookup
    
    def vocab_size(self, include_low_frequency_tokens=False):
        if not include_low_frequency_tokens:
            return self._num_words-1
        else:
            return len(self._token_lookup.word_index)-1