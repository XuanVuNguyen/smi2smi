import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np
import os

def fit_and_save(model, train_params, input_seqs, output_seq, valid_input_seqs, valid_output_seq):
    '''
    Args:
     input_seqs: a list of train_source and train_target_input, each is a tensor(batch_size, max_len).
     output_seq: tensor(batch_size, max_len); train_target_output.
     train_params: a dict of training parameters.
    '''

    opt = keras.optimizers.get(train_params['optimizer'])
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    steps_per_epoch = output_seq.shape[0]//train_params['batch_size'] + int(output_seq.shape[0]%train_params['batch_size']>0)
    checkpoint_path = os.path.join(train_params['save_dir'], 'cp-{epoch:05d}.h5')
    if train_params['initial_epoch'] > 0:
        model.load_weights(checkpoint_path.format(epoch=train_params['initial_epoch'])).expect_partial()
    else:
        model.save_weights(checkpoint_path.format(epoch=0))
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_freq=train_params['save_freq']*steps_per_epoch
        )
    tensorboard_callback = LRTensorBoard(log_dir=train_params['save_dir'], histogram_freq=train_params['save_freq'])
    
    if train_params['exponential_decay_lr'] and not train_params['cyclic_lr'] and not train_params['transformer_lr']:
        learning_rate_decay = [LearningRateExponentialDecay(
            init_learning_rate=train_params['exponential_decay_lr_config']['initial_learning_rate'], 
            decay_rate=train_params['exponential_decay_lr_config']['decay_rate'], 
            decay_steps=train_params['exponential_decay_lr_config']['decay_steps'], 
            min_learning_rate=train_params['exponential_decay_lr_config']['min_learning_rate'],
            staircase=train_params['exponential_decay_lr_config']['staircase']
        )]
    elif not train_params['exponential_decay_lr']:
        learning_rate_decay = []

    if train_params['cyclic_lr'] and not train_params['exponential_decay_lr'] and not train_params['transformer_lr']:
        cyclic_lr = [CustomCyclicLR(
            max_lr=train_params['cyclic_lr_config']['max_lr'],
            iters_per_cycle=train_params['cyclic_lr_config']['epochs_per_cycle']*steps_per_epoch,
            warm_up=train_params['cyclic_lr_config']['warm_up_epochs']*steps_per_epoch,
        )]
    elif not train_params['cyclic_lr']:
        cyclic_lr = []
        
    if train_params['transformer_lr'] and not train_params['exponential_decay_lr'] and not train_params['cyclic_lr']:
        transformer_lr = [TransformerLR(
            warm_up=train_params['transformer_lr_config']['warm_up_epochs']*steps_per_epoch,
            model_dim=train_params['transformer_lr_config']['model_dim'])]
    elif not train_params['transformer_lr']:
        transformer_lr = []
    
    callbacks = [checkpoint_callback, tensorboard_callback] + learning_rate_decay + cyclic_lr + transformer_lr
        
    train_history = model.fit(
        x=input_seqs,
        y=output_seq,
        batch_size=train_params['batch_size'], #last batch included
        epochs=train_params['epochs'],
        validation_data=(valid_input_seqs, valid_output_seq),
        validation_freq=train_params['save_freq'],
        callbacks=callbacks,
        shuffle=True, 
        initial_epoch=train_params['initial_epoch']
        )
    return train_history

# class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
#     @tf.function
#     def call(self, y_true, y_pred):
#         mask = tf.cast(y_true!=0, tf.float32)
#         losses = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
#         losses = losses*mask
#         return tf.reduce_sum(losses)/tf.reduce_sum(mask)

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, histogram_freq, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, histogram_freq=histogram_freq, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
        
class CustomCyclicLR(Callback):
    '''
   
    '''
    def __init__(self, max_lr=0.0012, iters_per_cycle=2000., warm_up=16000):
        super(CustomCyclicLR, self).__init__()
        self.max_lr = max_lr
        self.warm_up=warm_up
        self.iters_per_cycle=iters_per_cycle
        self.clr_iterations = 0.
        self.trn_iterations = 0. # one-based numbering, while tensorboard is zero-based numbering
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        if self.trn_iterations>self.iters_per_cycle:
            u = self.warm_up + np.mod(self.trn_iterations-1, self.iters_per_cycle)
        else:
            u = self.trn_iterations
        return self.max_lr*self.warm_up*np.minimum(1., u/self.warm_up)/np.maximum(u, self.warm_up)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.trn_iterations == 0:
            K.set_value(self.model.optimizer.lr, 0.)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        self.trn_iterations += 1
        self.clr_iterations += 1

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

class LearningRateExponentialDecay(Callback):
    def __init__(self, init_learning_rate, decay_rate, decay_steps, min_learning_rate, staircase):
        super(LearningRateExponentialDecay, self).__init__()
        self.init_learning_rate = init_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_learning_rate = min_learning_rate
        self.staircase = staircase
        self.history = {}
        self.trn_iterations = 0
    
    def lr_decay(self):
        if self.staircase:
            decayed_learning_rate = np.maximum(
                self.init_learning_rate*self.decay_rate**(self.trn_iterations//self.decay_steps),
                self.min_learning_rate)
        elif not self.staircase:
            decayed_learning_rate = np.maximum(
                self.init_learning_rate*self.decay_rate**(self.trn_iterations/self.decay_steps),
                self.min_learning_rate)
        return decayed_learning_rate
    
    def on_train_begin(self, logs={}):
        logs = logs or {}
        
        if self.trn_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.init_learning_rate)
        else:
            K.set_value(self.model.optimizer.lr, self.lr_decay())
    
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        self.trn_iterations += 1

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.lr_decay())

class TransformerLR(Callback):
    def __init__(self, 
                 warm_up,
                 model_dim):
        super().__init__()
        self.warm_up = warm_up
        self.model_dim = model_dim
        
        self.history = {}
        self.iterations = 0
    
    def calculate_lr(self):
        lr = np.power(self.model_dim, -0.5) * np.minimum(np.power(self.iterations, -0.5), 
                                                         self.iterations*np.power(self.warm_up, -1.5))
        return lr
    
    def on_train_begin(self, logs=None):
        logs = logs or {}
        
        if self.iterations==0:
            K.set_value(self.model.optimizer.lr, 0.)
        else:
            K.set_value(self.model.optimizer.lr, self.calculate_lr())
            
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iterations)
        
        self.iterations += 1
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.calculate_lr())