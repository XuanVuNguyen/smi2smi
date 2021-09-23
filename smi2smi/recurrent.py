import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from opennmt.utils import dynamic_decode, BeamSearch
import os

class MultiChannelConv1D(layers.Layer):
    def __init__(self,
                 kernel_sizes,
                 n_filters,
                 residual_connect,
                 layer_norm,
                 name='multi_channel_conv1d', **kwargs):
        super().__init__(name=name, **kwargs)
        self._residual_connect=residual_connect
        self.cnns = [layers.Conv1D(
            filters = n_filters,
            kernel_size = size,
            padding='same',
            activation='tanh',
            name=f'cnn_size_{size}') for size in kernel_sizes]
        if layer_norm:
            self._layer_norm = layers.LayerNormalization(name='layer_norm')
        else:
            self._layer_norm = None
            
        self.supports_masking = True
    def call(self, inputs, mask=None):
        inputs_features = [cnn(inputs) for cnn in self.cnns]
        all_features = layers.Add()(inputs_features)
        if self._residual_connect:
            all_features = layers.Add()([all_features, inputs])
        if self._layer_norm is not None:
            all_features = self._layer_norm(all_features)
        if mask is not None:
            mask_ = tf.expand_dims(tf.cast(mask, tf.float32), -1)
            all_features = all_features*mask_
        return all_features

class Bridge(layers.Layer):
    def __init__(self, 
                 bidirectional_encoding,
                 decoder_depth,
                 state_bridge,
                 name='bridge', **kwargs):
        super().__init__(name=name, **kwargs)
        self._bidirectional_encoding = bidirectional_encoding
        self._decoder_depth = decoder_depth
        self._state_bridge = state_bridge
        
    def __call__(self, encoder_state):
        if self._bidirectional_encoding:
            encoder_state = encoder_state[:(len(encoder_state)//2)]
        if self._state_bridge=='pass_through':
            decoder_init_state = encoder_state
        elif self._state_bridge=='last_of_encoder':
            decoder_init_state = [encoder_state[-1]]+ \
                [[tf.zeros((tf.shape(encoder_state[0][0])))]*2]*(self._decoder_depth-1)
        elif self._state_bridge=='zero':
            decoder_init_state = [[tf.zeros((tf.shape(encoder_state[0][0])))]*2]*\
                self._decoder_depth
        return decoder_init_state

class Attention(layers.Layer):
    def __init__(self,
                 attention_mechanism,
                 attention_dim,
                 return_attention_scores=False,
                 name='attention', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self._return_att_scores = return_attention_scores
        self._W1 = layers.Dense(attention_dim, use_bias=False, name='w1')
        self._W2 = layers.Dense(attention_dim, use_bias=False, name='w2')
        if attention_mechanism=='Bahdanau':
            self._attention = layers.AdditiveAttention(use_scale=True, name='bahdanau_attention')
        if attention_mechanism=='Luong':
            self._attention = layers.Attention(use_scale=False, name='luong_attention')
            
        self.supports_masking = True
            
    def call(self, query_and_values):
        query, values = query_and_values
        query_with_time = tf.expand_dims(query, 1)
        attention_output = self._attention(
            [self._W1(query_with_time), self._W2(values)], 
            return_attention_scores=self._return_att_scores)
        
        if type(attention_output) is tuple:
            context_vector = tf.squeeze(attention_output[0], [1])
            attention_scores = tf.squeeze(attention_output[1], [1])
            output = [context_vector, attention_scores]
        else:
            context_vector = tf.squeeze(attention_output, [1])
            output = context_vector
            
        return output

class DropoutRNN(layers.Layer):
    def __init__(self,
                 cell, 
                 return_sequences=True, 
                 return_state=True, 
                 go_backwards=False, 
                 name='dropout_rnn', **kwargs):
        super().__init__(name=name, **kwargs)
        #self.input_spec = layers.InputSpec(dtype=tf.float32, ndim=3)
        self.cell = cell
        self.return_sequences=return_sequences
        self.return_state=return_state
        self.go_backwards = go_backwards
        self.rnn = layers.RNN(cell=self.cell,
                              return_sequences=return_sequences, 
                              return_state=return_state, 
                              go_backwards=go_backwards,
                              name='rnn'
                              )
        self.supports_masking = True
        
    def call(self, inputs, initial_state=None, constants=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        outputs = self.rnn(inputs, initial_state=initial_state, constants=constants)
        if type(outputs) is list:
            outputs = tuple([outputs[0], outputs[1:]])
        return outputs
    
class BidirectionalRNN(layers.Layer):
    def __init__(self, fw_cell, bw_cell, name='bidirectional_rnn', **kwargs):
        super().__init__(name=name, **kwargs)
        self.fw_layer = DropoutRNN(fw_cell, go_backwards=False, name='fw_rnn')
        self.bw_layer = DropoutRNN(bw_cell, go_backwards=True, name='bw_rnn')
    def call(self, inputs):
        fw_seq, fw_state = self.fw_layer(inputs)
        bw_seq, bw_state = self.bw_layer(inputs)
        bw_seq = tf.reverse(bw_seq, axis=[1])
        output = layers.Concatenate(axis=-1)([fw_seq, bw_seq])
        out_state = fw_state + bw_state 
        return output, out_state
    
class StackedLSTMCell(layers.AbstractRNNCell):
    def __init__(self, 
                 hidden_dim, 
                 depth,
                 dropout_rate_input,
                 dropout_rate_output,
                 residual_connect,
                 name='multi_lstm', **kwargs):
        super(StackedLSTMCell, self).__init__(name=name, **kwargs)
        self._residue_type = residual_connect
        self.units = [layers.LSTMCell(
            hidden_dim,
            dropout=dropout_rate_input,
            recurrent_dropout=dropout_rate_output,
            unit_forget_bias=True,
            name=f'LSTM_{i}')
            for i in range(depth)]
    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self.units)
    
    @property
    def output_size(self):
        return self.units[-1].output_size
    
    def get_config(self):
        return {'params': self.params, 'sub_module':self.sub_module}

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [cell.get_initial_state(inputs, batch_size, dtype) \
                for cell in self.units]
        
    def call(self, token, inp_state):
        '''
          hidden_state: nested list of tensors
        '''
        cur_inp = token
        all_inputs_sum = cur_inp
        out_state = []
        for i, cell in enumerate(self.units):
            h, h_and_c = cell(cur_inp, inp_state[i])    
            if self._residue_type=='Res':
                cur_inp = layers.Add()([cur_inp, h])
            elif self._residue_type=='ResD':
                cur_inp = layers.Add()([all_inputs_sum, h])
                all_inputs_sum = layers.Add()([all_inputs_sum, cur_inp])
            elif self._residue_type is None:
                cur_inp = h
            out_state.append(h_and_c)
        return cur_inp, out_state
    
    def reset_dropout_mask(self):
        for unit in self.units:
            unit.reset_dropout_mask()
    def reset_recurrent_dropout_mask(self):
        for unit in self.units:
            unit.reset_recurrent_dropout_mask()
        
class DecoderCell(layers.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 decoder_rnn_config,
                 attention_config,
                 attention_scores_as_output=False,
                 name='decoder_cell', **kwargs):
        super().__init__(name=name, **kwargs)
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._decoder_rnn_config = decoder_rnn_config
        self._attention_config = attention_config
        self._attention_scores_as_output = attention_scores_as_output
        self._name = name
        
        self._rnn_cell = StackedLSTMCell(
            hidden_dim=decoder_rnn_config['hidden_dim'],
            depth=decoder_rnn_config['depth'],
            dropout_rate_input=decoder_rnn_config['dropout_rate_input'],
            dropout_rate_output=decoder_rnn_config['dropout_rate_output'],
            residual_connect=decoder_rnn_config['residual_connect'])
        
        lstm_input_dim = attention_config['attention_dim']*attention_config['append_context'] + embedding_dim
        if self._rnn_cell._residue_type is not None and lstm_input_dim!=hidden_dim:
            self._input_transform = layers.Dense(self.cell.output_size, name='input_transform')
        else:
            self._input_transform = None
            
        if attention_config['attention_mechanism'] is not None:
            self._attention = Attention(
                attention_mechanism=attention_config['attention_mechanism'],
                attention_dim=attention_config['attention_dim'],
                return_attention_scores=attention_scores_as_output,
                name='attention')
          
            self._attention_mix = layers.Dense(decoder_rnn_config['hidden_dim'],
                                              activation='tanh',
                                              name='attention_mix')
        else:
            self._attention = None
            
        self._logit_dense = layers.Dense(vocab_size,
                                          name='softmax_dense')
    
    def call(self, token, cache, constants=None):
        '''
        Args:
          token: tensor(batch_size, vocab_sizes)
          cache: a list of rnn states and maybe context vector.
          constants: None or a list of tensors.
        Outputs:
          prob_dist: tensor(batch_size, vocab_size)
          states: tuple([[tensor(batch_size, hidden_dim), tensor(batch_size, hidden_dim)]]*depth)
        '''
        if len(cache)==1: # append_context is False
            states, = cache
        if len(cache)==2: # append_context is True
            states, context_vector = cache
        rnn_input = token
        
        if self._attention is not None and constants is not None: 
            keys, = constants
            if self._attention_config['append_context']:
                rnn_input = tf.concat([context_vector, rnn_input], axis=-1)
                if self._input_transform is not None:
                    rnn_input = self._input_transform(rnn_input)
        #query shape == (batch_size, hidden_dim)
        query, next_states = self._rnn_cell(rnn_input, states)
        next_cache = [next_states]
        #context_vector shape == (batch_size, hidden_dim)
        if self._attention is not None:
            att_output = self._attention([query, keys])
                
            next_context = att_output[0] if isinstance(att_output, tuple) else att_output
            if self._attention_config['append_context']:
                next_cache += [next_context]
            if isinstance(att_output, tuple):
                att_scores = att_output[1]
                return att_scores, next_cache
            else:              
                query = tf.concat([next_context, query], axis=-1)
                query = self._attention_mix(query)

        logits = self._logit_dense(query)
        return logits, next_cache
    
    @property
    def state_size(self):
        if self._attention_config['append_context'] and self._attention_config['attention_mechanism'] is not None:
            return [list(self._rnn_cell.state_size),]+[self._attention_config['attention_dim'],]
        else:
            return [list(self._rnn_cell.state_size),]
    @property
    def output_size(self):
        if self._attention_scores_as_output:
            return self._attention_config['attention_dim']
        else:
            return self._vocab_size
    
    def get_config(self):
        return {'vocab_size':self._vocab_size,
                'embedding_dim':self._embedding_dim,
                'decoder_rnn_config':self._decoder_rnn_config,
                'attention_config':self._attention_config,
                'attention_scores_as_output':self._attention_scores_as_output}
    
    def reset_dropout_mask(self):
        self._rnn_cell.reset_dropout_mask()
    def reset_recurrent_dropout_mask(self):
        self._rnn_cell.reset_recurrent_dropout_mask()
    
class Decoder(layers.Layer):
    def __init__(self,
                 vocab_size,
                 embedding,
                 decoder_rnn_config,
                 attention_config,
                 attention_scores_as_output=False,
                 name='decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self._attention_config = attention_config
        self._attention_dim = attention_config['attention_dim']*bool(attention_config['attention_mechanism'])
        if isinstance(embedding, int):
            self._embedding = layers.Embedding(vocab_size, embedding,
                                               mask_zero=True,
                                               name='decoder_embedding')
            embedding_dim = embedding
        elif isinstance(embedding, layers.Layer):
            self._embedding = embedding
            embedding_dim = self._embedding.output_dim
            
        self._decoder_cell = DecoderCell(
            vocab_size,
            embedding_dim,
            decoder_rnn_config,
            attention_config,
            attention_scores_as_output=attention_scores_as_output)
        
        self._dynamic_decoder = DropoutRNN(
            self._decoder_cell,
            return_state=False,
            name='dynamic_decoder')
    
    def call(self, inputs, initial_cache, encoder_hidden_seq=None):
        embedded_inputs = self._embedding(inputs)
#         initial_cache = [init_state, ]
        logits = self._dynamic_decoder(embedded_inputs,
                                    initial_state=initial_cache,
                                      constants=encoder_hidden_seq)
        return logits
    
    def get_initial_context_vector(self, batch_size):
        return tf.zeros((batch_size, self._attention_dim))

class InferenceDecoder(Decoder):
    def __init__(self,
                 vocab_size,
                 embedding,
                 decoder_rnn_config,
                 attention_config,
                 name='decoder'):
        super().__init__(vocab_size=vocab_size,
                         embedding=embedding,
                         decoder_rnn_config=decoder_rnn_config,
                         attention_config=attention_config,
                         name=name)
    def call(self, token, step, cache):
        embedded_token = self._embedding(token)
        
        if len(cache)==3:
            states, context_vector, encoder_seq = cache
            constants = [encoder_seq]
            states_and_maybe_context = [states, context_vector]
            
        else:
            if len(cache)==2:
                states, encoder_seq = cache
                constants = [encoder_seq]
            if len(cache)==1:
                states, = cache
                constants = None
            states_and_maybe_context = [states]
            
        logits, next_states_and_maybe_context = self._decoder_cell(embedded_token, states_and_maybe_context,
                                                                   constants)

        if len(cache)>=1:
            next_cache = next_states_and_maybe_context + [encoder_seq]
            
        return logits, next_cache

class Encoder(layers.Layer):
    def __init__(self, 
                 vocab_size,
                 embedding_dim,
                 cnn_config,
                 encoder_rnn_config,
                 name='encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self._embedding = layers.Embedding(vocab_size, embedding_dim,
                                           mask_zero=True)
        
        self._inception_layers = [MultiChannelConv1D(
            kernel_sizes=cnn_config['kernel_sizes'],
            n_filters=cnn_config['n_filters'],
            residual_connect=cnn_config['residual_connect'],
            layer_norm=cnn_config['layer_norm'],
            name=f'inception_{i}') for i in range(cnn_config['depth'])]
        
        cell = StackedLSTMCell(
            hidden_dim=encoder_rnn_config['hidden_dim'], 
            depth=encoder_rnn_config['depth'],
            dropout_rate_input=encoder_rnn_config['dropout_rate_input'],
            dropout_rate_output=encoder_rnn_config['dropout_rate_output'],
            residual_connect=encoder_rnn_config['residual_connect'])
        
        if not encoder_rnn_config['bidirectional_encoding']:
            self._rnn = DropoutRNN(cell)
        else:
            bw_cell = StackedLSTMCell(
                hidden_dim=encoder_rnn_config['hidden_dim'], 
                depth=encoder_rnn_config['depth'],
                dropout_rate_input=encoder_rnn_config['dropout_rate_input'],
                dropout_rate_output=encoder_rnn_config['dropout_rate_output'],
                residual_connect=encoder_rnn_config['residual_connect'],
                name='bw_multi_lstm')
            self._rnn = BidirectionalRNN(cell, bw_cell)
    
    def call(self, inputs):
        embedded_inputs = self._embedding(inputs)
        for inception in self._inception_layers:
            embedded_inputs = inception(embedded_inputs)
        outputs, states = self._rnn(embedded_inputs)
        return outputs, states
    
    @property
    def embedding_layer(self):
        return self._embedding

def build_model(model_config, attention_scores_as_output=False):
    '''
    Used for training only.
    '''
    source_seq = keras.Input((None, ), dtype='int32', name='source_sequence')
    target_seq_inp = keras.Input((None, ), dtype='int32', name='target_sequence_input')
    if model_config['source_reverse']:
        pre_embed_seq = tf.reverse(source_seq, [1])
    elif not model_config['source_reverse']:
        pre_embed_seq = source_seq

    encoder = Encoder(
        vocab_size=model_config['vocab_size'],
        embedding_dim=model_config['embedding_dim'],
        cnn_config=model_config['cnn_config'],
        encoder_rnn_config=model_config['encoder_rnn_config'])
    
    bridge = Bridge(
        bidirectional_encoding=model_config['encoder_rnn_config']['bidirectional_encoding'],
        decoder_depth=model_config['decoder_rnn_config']['depth'],
        state_bridge=model_config['state_bridge'])
    
    if model_config['embedding_share']:
        decoder_embedding = encoder.embedding_layer
    elif not model_config['embedding_share']:
        decoder_embedding = model_config['embedding_dim']
        
    decoder = Decoder(
        vocab_size=model_config['vocab_size'],
        embedding=decoder_embedding,
        decoder_rnn_config=model_config['decoder_rnn_config'],
        attention_config=model_config['attention_config'],
        attention_scores_as_output=attention_scores_as_output)
    
    hidden_seq, encoder_state = encoder(pre_embed_seq)
    decoder_init_state = bridge(encoder_state)
    encoder_hidden_seq = [hidden_seq]
    if model_config['attention_config']['attention_mechanism'] is None:
        decoder_init_cache = [decoder_init_state]
        encoder_hidden_seq = None
    else:
        decoder_init_cache = [decoder_init_state, 
                              decoder.get_initial_context_vector(tf.shape(source_seq)[0])]
    logits_or_att_scores = decoder(target_seq_inp, 
                                   decoder_init_cache,
                                   encoder_hidden_seq=encoder_hidden_seq)  
    return keras.Model([source_seq, target_seq_inp], logits_or_att_scores, name='model')

class InferenceModel(keras.Model):
    def __init__(self, 
                 model_config, 
                 infer_config, 
                 name='infer_model', **kwargs):
        super().__init__(name=name, **kwargs)
        self._model_config = model_config
        self._infer_config = infer_config
        self._beam_width = infer_config['beam_width']
        self._max_decode_length = infer_config['max_decode_length']
        self._length_penalty = infer_config['length_penalty']

        self._encoder = Encoder(
            vocab_size=model_config['vocab_size'],
            embedding_dim=model_config['embedding_dim'],
            cnn_config=model_config['cnn_config'],
            encoder_rnn_config=model_config['encoder_rnn_config'])
    
        if model_config['embedding_share']:
            decoder_embedding = self._encoder.embedding_layer
        elif not model_config['embedding_share']:
            decoder_embedding = model_config['embedding_dim']
            
        self._decoder = InferenceDecoder(
            vocab_size=model_config['vocab_size'],
            embedding=decoder_embedding,
            decoder_rnn_config=model_config['decoder_rnn_config'],
            attention_config=model_config['attention_config'])
        self._bridge = Bridge(
            bidirectional_encoding=model_config['encoder_rnn_config']['bidirectional_encoding'],
            decoder_depth=model_config['decoder_rnn_config']['depth'],
            state_bridge=model_config['state_bridge'])
        
    def call(self, source_seq):
        '''
        Args:
          source_seq: tensor(1, max_len)
        Returns:
          beam_search_sequence: tensor(batch_size, max_decode_len, beam_width)
          beam_search_output
        '''
        batch_size = tf.shape(source_seq)[0]
        if self._model_config['source_reverse']:
            inputs = tf.reverse(source_seq, [1])
        else:
            inputs = source_seq

        encoder_hidden_seq, encoder_states = self._encoder(inputs)
        encoder_states = [[self._tile_batch(state, self._beam_width) \
                           for state in h_and_c] for h_and_c in encoder_states]
        
        decoder_init_states = self._bridge(encoder_states)
        
        decoder_init_cache = [decoder_init_states]
        
        if self._model_config['attention_config']['attention_mechanism'] is not None:
            encoder_hidden_seq = self._tile_batch(encoder_hidden_seq, self._beam_width)
            decoder_init_cache = [decoder_init_states, encoder_hidden_seq]
            if self._model_config['attention_config']['append_context']:
                init_context = self._decoder.get_initial_context_vector(batch_size*self._beam_width)
                decoder_init_cache = [decoder_init_states, init_context, encoder_hidden_seq]
                
        start_tokens = tf.fill([batch_size], 1)
        end_token = 2
        
        decoding_strategy = BeamSearch(beam_size=self._beam_width,
                                       length_penalty=self._length_penalty)
        
        outputs = dynamic_decode(
            symbols_to_logits_fn=self._decoder,
            start_ids=start_tokens,
            end_id=end_token,
            initial_state=decoder_init_cache,
            decoding_strategy=decoding_strategy,
            maximum_iterations=self._max_decode_length)
        
        return outputs

    def _tile_batch(self, tensor, multiplier):
        return tf.repeat(tensor, 
                         tf.ones((tf.shape(tensor)[0],), dtype=tf.int32)*multiplier,
                         axis=0)
    
    def build_and_load_weights(self):
        model_dim = self._model_config['encoder_rnn_config']['hidden_dim']
        attention_dim = self._model_config['attention_config']['attention_dim']
        encoder_output_dim = self._model_config['encoder_rnn_config']['hidden_dim'] \
            *(1 + self._model_config['encoder_rnn_config']['bidirectional_encoding'])
        
        _ = self.layers[0](tf.zeros((1, 1)))
        _ = self.layers[1](tf.zeros((1, )), 0, 
                                [[[tf.zeros((1, self._model_config['encoder_rnn_config']['hidden_dim']))]*2], 
                                tf.zeros((1, self._model_config['attention_config']['attention_dim'])),
                                tf.zeros((1, 1, encoder_output_dim))])
        
        dummy_model = build_model(self._model_config)
        weight_path = os.path.join(self._infer_config['model_dir'], 
                                   'cp-{:05d}.h5'.format(self._infer_config['epoch_number']))
        dummy_model.load_weights(weight_path)
        self.set_weights(dummy_model.get_weights())
        
    @property    
    def configs(self):
        return {'model_config':self._model_config, 'infer_config':self._infer_config}