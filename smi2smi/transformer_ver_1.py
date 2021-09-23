import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import os

from opennmt.utils import dynamic_decode, BeamSearch



def positional_embedding(seq_len, model_dim):
    def _get_angles(pos, i, model_dim):
        return tf.cast(pos, tf.float32) \
            /tf.math.pow(10000., tf.cast(2*(i//2), tf.float32) / tf.cast(model_dim, tf.float32))
    
    angle_rads = _get_angles(tf.expand_dims(tf.range(seq_len), -1),
                                 tf.expand_dims(tf.range(model_dim), 0),
                                 model_dim) #shape==(seq_len, model_dim)

    sin_indices = tf.range(0, tf.shape(angle_rads)[1], 2)
    cos_indices = tf.range(1, tf.shape(angle_rads)[1], 2)

    sin_values = tf.math.sin(tf.gather(angle_rads, sin_indices, axis=-1))
    cos_values = tf.math.cos(tf.gather(angle_rads, cos_indices, axis=-1))

    sin_values = tf.expand_dims(sin_values, -1)
    cos_values = tf.expand_dims(cos_values, -1)

    pos_emb = tf.concat([sin_values, cos_values], -1)
    return tf.reshape(pos_emb, (1, seq_len, model_dim))

class TokAndPosEmbedding(layers.Layer):
    def __init__(self,
                 input_dim,
                 model_dim,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_dim = model_dim
        
        self.tok_emb = layers.Embedding(input_dim=input_dim,
                                        output_dim=model_dim,
                                        name='embedding')
        self.supports_masking = True
        
    def call(self, inputs):
        embedded_inp = self.tok_emb(inputs) #shape==(B, len, feat)
        max_seq_len = tf.shape(inputs)[1]
        pos_vec = positional_embedding(max_seq_len, self.model_dim) #shape==(1, len, feat)
        return embedded_inp + pos_vec
    
    @property
    def token_embedding_layer(self):
        return self.tok_emb
    
    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

class MultiHeadAtt(layers.Layer):
    def __init__(self,
                 num_heads,
                 key_dim,
                 value_dim,
                 dropout,
                 att_output_shape=None, 
                 name='multihead_attention',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.multihead_att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dropout=dropout,
            output_shape=att_output_shape,
            name='multihead_att')
        self.layernorm = layers.LayerNormalization(name='layer_norm')
        
        self.supports_masking = True
        
    def call(self, queries, keys, values, 
             attention_mask=None):
        outputs = self.multihead_att(query=queries, key=keys, value=values,
                                     attention_mask=attention_mask)
        outputs = self.layernorm(layers.Add()([queries, outputs]))
        return outputs
    
class FeedForward(layers.Layer):
    def __init__(self, 
                 dense_1_units,
                 dense_2_units,
                 dropout,
                 name='feed_forward',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.ff1 = layers.Dense(dense_1_units, activation='relu', name='dense_0')
        self.ff2 = layers.Dense(dense_2_units, name='dense_1')
        self.dropout = layers.Dropout(rate=dropout)
        self.layernorm = layers.LayerNormalization(name='layer_norm')
        
        self.supports_masking = True
        
    def call(self, inputs):
        outputs = self.ff1(inputs)
        outputs = self.ff2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layernorm(layers.Add()([inputs, outputs]))
        return outputs
    
class EncoderBlock(layers.Layer):
    def __init__(self, 
                 block_config,
                 name='encoder_block',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.self_att = MultiHeadAtt(
            num_heads=block_config['num_heads'],
            key_dim=block_config['key_dim'],
            value_dim=block_config['value_dim'],
            dropout=block_config['dropout'],
            att_output_shape=None)
        
        self.ff = FeedForward(
            dense_1_units=block_config['ff_inner_dim'],
            dense_2_units=block_config['model_dim'],
            dropout=block_config['dropout'])
        
    def call(self, inputs):
        outputs = self.self_att(inputs, inputs, inputs)
        outputs = self.ff(outputs)
        return outputs
    
class DecoderBlock(layers.Layer):
    def __init__(self,
                 block_config,
                 name='decoder_block',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.self_att = MultiHeadAtt(
            num_heads=block_config['num_heads'],
            key_dim=block_config['key_dim'],
            value_dim=block_config['value_dim'],
            dropout=block_config['dropout'],
            att_output_shape=None,
            name='self_attention')

        self.cross_att = MultiHeadAtt(
            num_heads=block_config['num_heads'],
            key_dim=block_config['key_dim'],
            value_dim=block_config['value_dim'],
            dropout=block_config['dropout'],
            att_output_shape=None,
            name='cross_attention')

        self.ff = FeedForward(
            dense_1_units=block_config['ff_inner_dim'],
            dense_2_units=block_config['model_dim'],
            dropout=block_config['dropout'])
        
        self.supports_masking=True
    
    def call(self, dec_inputs_and_enc_outputs, mask=None):
        dec_inputs, enc_outputs = dec_inputs_and_enc_outputs
        if mask is not None:
            dec_inp_len_mask = mask[0]
        else:
            dec_inp_len_mask = None
        future_mask = self.get_future_mask(dec_inputs, dec_inp_len_mask)
        outputs = self.self_att(dec_inputs, dec_inputs, dec_inputs,
                                attention_mask=future_mask)
        outputs = self.cross_att(outputs, enc_outputs, enc_outputs)
        outputs = self.ff(outputs)
        return outputs
    
    def get_future_mask(self, inputs, len_mask=None):
        
        future_mask_shape = (tf.shape(inputs)[0], 
                             tf.shape(inputs)[1], 
                             tf.shape(inputs)[1])    
        future_mask = tf.linalg.band_part(tf.ones(future_mask_shape), -1, 0)
        if len_mask is not None:
            future_mask = future_mask * tf.cast(tf.expand_dims(len_mask, -1), tf.float32)
        return tf.cast(future_mask, tf.bool)
    
class Encoder(layers.Layer):
    def __init__(self, 
                 encoder_depth,
                 vocab_size,
                 block_config,
                 name='encoder',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.embedding = TokAndPosEmbedding(vocab_size, block_config['model_dim'])
        
        self.blocks = [EncoderBlock(
            block_config,
            name=f'encoder_block_{i}') for i in range(encoder_depth)]
        
        self.supports_masking = True
        
    def call(self, inputs):
        embedded_inputs = self.embedding(inputs)
        cur_inp = embedded_inputs
        for block in self.blocks:
            block_outputs = block(cur_inp)
            cur_inp = block_outputs
        return cur_inp
    
    @property
    def embedding_layer(self):
        return self.embedding
    
class ApplySoftmaxTemp(layers.Layer):
    def __init__(self, temp, **kwargs):
        super().__init__(**kwargs)
        self.temp = temp
        self.supports_masking = True
    
    def call(self, inputs):
        return inputs/self.temp
    
class Decoder(layers.Layer):
    def __init__(self, 
                 decoder_depth,
                 vocab_size,
                 block_config,
                 softmax_temperature=1.,
                 embedding_layer=None,
                 name='decoder',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if embedding_layer is None:
            self.embedding = TokAndPosEmbedding(vocab_size, block_config['model_dim'])
        else:
            self.embedding = embedding_layer
        
        self.blocks = [DecoderBlock(
            block_config,
            name=f'decoder_block_{i}') for i in range(decoder_depth)]
        
        self.logit_dense = layers.Dense(vocab_size, 
                                        name='logit_dense')
        
        self.apply_temperature = ApplySoftmaxTemp(softmax_temperature)
        
        self.supports_masking = True
    
    def call(self, dec_inputs, enc_outputs):
        embedded_dec_inputs = self.embedding(dec_inputs)
        
        cur_inp = embedded_dec_inputs
        for block in self.blocks:
            block_outputs = block([cur_inp, enc_outputs])
            cur_inp = block_outputs
        
        logits = self.logit_dense(cur_inp)
        logits = self.apply_temperature(logits)
        return logits
    
class InferenceDecoderBlock(DecoderBlock):
    def __init__(self,
                 block_config,
                 name='decoder_block',
                 **kwargs):
        super().__init__(block_config, name, **kwargs)
        self.block_config = block_config
        
    def call(self, token, step, cache):
        '''
         token (B, 1, feats)
         time (int)
         '''
        token_with_time = tf.expand_dims(token, 1)
        enc_outputs = cache[0]
        cur_seq = cache[1]
        
        # time_step: int
        # latest_seq: (B, max_len, feats)
        # enc_outputs: (B, len, feats)
        updated_seq = self._update_seq(token, cur_seq, step)
        outputs = self.self_att(token_with_time, updated_seq[:, :(step+1)], updated_seq[:, :(step+1)])
        outputs = self.cross_att(outputs, enc_outputs, enc_outputs)
        outputs = self.ff(outputs)
        return tf.squeeze(outputs, 1), [enc_outputs, updated_seq]
    
    def _update_seq(self, token, cur_seq, step):
        token_list = tf.unstack(cur_seq, axis=1)
        token_list[step] = token
        return tf.stack(token_list, axis=1)
    
class InferenceDecoderCell(layers.Layer):
    def __init__(self, 
                 decoder_depth,
                 vocab_size,
                 block_config,
                 max_decode_length,
                 softmax_temperature=1.,
                 embedding_layer=None,
                 name='decoder',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.decoder_depth = decoder_depth
        self.vocab_size = vocab_size
        self.block_config = block_config
        self.max_decode_length = max_decode_length
        
        if embedding_layer is not None:
            self.embedding_layer = embedding_layer
        else:
            self.embedding_layer = layers.Embedding(vocab_size, 
                                                    block_config['model_dim'],
                                                    name='embedding')
        self.positional_vectors = positional_embedding(max_decode_length, 
                                                      block_config['model_dim']) #shape=(1, len, dim)
        
        self.blocks = [InferenceDecoderBlock(
            block_config, 
            name=f'decoder_block_{i}') for i in range(decoder_depth)]
        
        self.logit_dense = layers.Dense(vocab_size,
#                                         activation='softmax',
                                        name='logit_dense')
        
        self.apply_temperature = ApplySoftmaxTemp(softmax_temperature)
        
    def call(self, token, step, cache):
        '''
        Args:
            token: (B, 1)
            states: [int, (B, max_len, feats), (B, T, feats)] 
        '''
        enc_outputs = cache[0]
        cur_seqs = cache[1]
        embedded_token = self.embedding_layer(token)
        pos_embedded_token = embedded_token + self.positional_vectors[:, step]
        cur_block_token = pos_embedded_token #shape==(batch, feats)

        updated_seqs = []
        for block, cur_seq in zip(self.blocks, cur_seqs):
            block_cache = [enc_outputs,
                           cur_seq]
            output, next_cache = block(cur_block_token,
                                       step,
                                       block_cache)
            updated_seqs.append(next_cache[1])
            cur_block_token = output
        
        logit = self.logit_dense(cur_block_token)
        logit = self.apply_temperature(logit)
        return logit, [enc_outputs,
                       updated_seqs]
    
def build_model(model_config):
    enc_inputs = keras.Input(shape=(None, ), dtype='int32', name='encoder_inputs')
    dec_inputs = keras.Input(shape=(None, ), dtype='int32', name='decoder_inputs')
    
    encoder = Encoder(encoder_depth=model_config['encoder_depth'],
                      vocab_size=model_config['vocab_size'],
                      block_config=model_config['block_config'])
    
    if model_config['embedding_share']:
        dec_embedding_layer = encoder.embedding_layer
    else:
        dec_embedding_layer = None
    decoder = Decoder(decoder_depth=model_config['decoder_depth'],
                      vocab_size=model_config['vocab_size'],
                      embedding_layer=dec_embedding_layer,
                      block_config=model_config['block_config'],
                      softmax_temperature=model_config['softmax_temperature'])
    
    enc_outputs = encoder(enc_inputs)
    
    logits = decoder(dec_inputs, enc_outputs)
    
    return keras.Model([enc_inputs, dec_inputs],
                       logits,
                       name='model')

class InferenceModel(keras.Model):
    def __init__(self,
                 model_config,
                 inference_config,
                 name='inference_model',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_config = model_config
        self.model_dim = model_config['block_config']['model_dim']
        self.decoder_depth = model_config['decoder_depth']
        
        self.beam_width = inference_config['beam_width']
        self.length_penalty = inference_config['length_penalty']
        self.max_decode_length = inference_config['max_decode_length']
        self.model_dir = inference_config['model_dir']
        self.epoch_num = inference_config['epoch_number']
                                          
        self.encoder = Encoder(
            encoder_depth=model_config['encoder_depth'],
            vocab_size=model_config['vocab_size'],
            block_config=model_config['block_config'])
        
        if model_config['embedding_share']:
            dec_embedding_layer = self.encoder.embedding_layer.token_embedding_layer
        else:
            dec_embedding_layer = None
        
        self.decoder_cell = InferenceDecoderCell(
            decoder_depth=model_config['decoder_depth'],
            vocab_size=model_config['vocab_size'],
            block_config=model_config['block_config'],
            max_decode_length=inference_config['max_decode_length'],
            softmax_temperature=model_config['softmax_temperature'],
            embedding_layer=dec_embedding_layer)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        enc_outputs = self.encoder(inputs)
        initial_cache = [self._tile_batch(enc_outputs, self.beam_width), 
                         [tf.zeros((batch_size*self.beam_width, self.max_decode_length, self.model_dim))]\
                         *self.decoder_depth]
        
        start_tokens = tf.fill([batch_size], 1)
        end_token = 2
        
        decoding_strategy = BeamSearch(beam_size=self.beam_width,
                                       length_penalty=self.length_penalty)
        
        outputs = dynamic_decode(
            symbols_to_logits_fn=self.decoder_cell,
            start_ids=start_tokens,
            end_id=end_token,
            initial_state=initial_cache,
            decoding_strategy=decoding_strategy,
            maximum_iterations=self.max_decode_length)
        
        return outputs
                                                          
    def _tile_batch(self, tensor, multiplier):
        return tf.repeat(tensor, 
                         tf.ones((tf.shape(tensor)[0],), dtype=tf.int32)*multiplier,
                         axis=0)
    
    def build_and_load_weights(self):
        _ = self.layers[0](tf.zeros((1, 1)))
        _ = self.layers[1](tf.zeros((1, )), 0, 
                                     [tf.zeros((1, 1, self.model_dim)), 
                                      [tf.zeros((1, 1, self.model_dim))]*self.decoder_depth])
        
        dummy_model = build_model(self.model_config)
        weight_path = os.path.join(self.model_dir, 'cp-{:05d}.h5'.format(self.epoch_num))
        dummy_model.load_weights(weight_path)
        self.set_weights(dummy_model.get_weights())