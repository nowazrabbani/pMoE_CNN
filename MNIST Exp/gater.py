# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:06:39 2022

@author: nowaz
"""

import tensorflow as tf

class gate(tf.keras.layers.Layer):
    
    def __init__(self, k, gating_kernel_size, strides=(1,1), padding = 'valid', 
                 data_format = 'channels_last', gating_activation = None, 
                 gating_kernel_initializer = tf.keras.initializers.RandomNormal, **kwargs):
        
        super(gate, self).__init__(**kwargs)
        self.k = k
        self.gating_kernel_size = gating_kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.gating_activation = tf.keras.activations.get(gating_activation) 
        self.gating_kernel_initializer = gating_kernel_initializer 
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)
        
    def build(self, input_shape):
        
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        input_dim = input_shape[channel_axis]
        
        
        gating_kernel_shape = self.gating_kernel_size + (input_dim, 1)
        
        self.gating_kernel = self.add_weight(shape=gating_kernel_shape,
                                      initializer=self.gating_kernel_initializer,
                                      name='gating_kernel')
    
    def call(self, inputs):
        
        gating_outputs = tf.keras.backend.conv2d(inputs, self.gating_kernel, strides=self.strides, 
                                  padding=self.padding,data_format=self.data_format)
        
        gating_outputs = tf.transpose(gating_outputs, perm=(0,3,1,2))
        
        x = tf.shape(gating_outputs)[2]
        
        y = tf.shape(gating_outputs)[3]
        
        gating_outputs = tf.reshape(gating_outputs,(tf.shape(gating_outputs)[0],tf.shape(gating_outputs)[1],
                                                    x*y))
        
        [values, indices] = tf.math.top_k(gating_outputs,k=self.k, sorted=False)
        
        values = self.gating_activation(values)
        
        indices = tf.reshape(indices,(tf.shape(indices)[0]*tf.shape(indices)[1],tf.shape(indices)[2]))
        
        values = tf.reshape(values, (tf.shape(values)[0]*tf.shape(values)[1], tf.shape(values)[2]))
        
        batch_t, k_t = tf.unstack(tf.shape(indices), num=2)
        
        n=tf.shape(gating_outputs)[2]
        
        indices_flat = tf.reshape(indices, [-1]) + tf.math.floordiv(tf.range(batch_t * k_t), k_t) * n
        
        ret_flat = tf.math.unsorted_segment_sum(tf.reshape(values, [-1]), indices_flat, batch_t * n)
        
        ret_rsh=tf.reshape(ret_flat, [batch_t, n])
        
        ret_rsh_3=tf.reshape(ret_rsh,(tf.shape(gating_outputs)[0],tf.shape(gating_outputs)[1],tf.shape(gating_outputs)[2]))
        
        new_gating_outputs = tf.reshape(ret_rsh_3,(tf.shape(ret_rsh_3)[0],tf.shape(ret_rsh_3)[1],x,y))
        
        new_gating_outputs = tf.transpose(new_gating_outputs, perm=(0,2,3,1))
        
        new_gating_outputs = tf.repeat(new_gating_outputs,tf.shape(self.gating_kernel)[0]*tf.shape(self.gating_kernel)[1]*tf.shape(self.gating_kernel)[2],axis=3)
        
        new_gating_outputs=tf.reshape(new_gating_outputs,(tf.shape(new_gating_outputs)[0],tf.shape(new_gating_outputs)[1],tf.shape(new_gating_outputs)[2],tf.shape(self.gating_kernel)[0],tf.shape(self.gating_kernel)[1],tf.shape(self.gating_kernel)[2]))
        
        new_gating_outputs=tf.transpose(new_gating_outputs,perm=(0,1,3,2,4,5))
        
        new_gating_outputs=tf.reshape(new_gating_outputs,(tf.shape(new_gating_outputs)[0],tf.shape(new_gating_outputs)[1]*tf.shape(new_gating_outputs)[2],tf.shape(new_gating_outputs)[3]*tf.shape(new_gating_outputs)[4],tf.shape(new_gating_outputs)[5]))
        
        outputs = inputs*new_gating_outputs
        
        return outputs
        
        
        