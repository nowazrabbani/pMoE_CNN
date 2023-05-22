
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from gater_2 import gate


initializer_gate=keras.initializers.RandomNormal(mean=0.0,stddev=0.0001)



def WideResnetBlock(x, channels, strides, channel_mismatch=False):
    
    identity = x
    
    out = layers.BatchNormalization()(x)
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=channels, kernel_size=3, strides=strides, padding='same')(out)
    
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding='same')(out)
    
    if channel_mismatch is not False:
        identity = layers.Conv2D(filters=channels, kernel_size=1, strides=strides, padding='valid')(identity)
    
    out = layers.Add()([identity, out])
    
    return out

def WideResnetGroup(x, num_blocks, channels, strides):
    
    x = WideResnetBlock(x=x, channels=channels, strides=strides, channel_mismatch=True)
    
    for _ in range(num_blocks - 1):
        x = WideResnetBlock(x=x, channels=channels, strides=(1, 1))
    
    return x

def WideResnet(x, num_blocks, k, num_classes=10):
    widths = [int(v * k) for v in (16, 32, 64)]
    
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(x)
    
    x = WideResnetGroup(x, num_blocks, widths[0], strides=(1, 1))
    
    x = WideResnetGroup(x, num_blocks, widths[1], strides=(2, 2))
    
    x = layers.BatchNormalization()(x)
    
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters=640, kernel_size=3, strides=2, padding='same')(x)
    
    x_1 = gate(16,(1,1),(1,1),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(x)
    x_2 = gate(16,(1,1),(1,1),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(x)
    x_3 = gate(16,(1,1),(1,1),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(x)
    x_4 = gate(16,(1,1),(1,1),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(x)
    
    x_1 = layers.BatchNormalization()(x_1)
    x_2 = layers.BatchNormalization()(x_2)
    x_3 = layers.BatchNormalization()(x_3)
    x_4 = layers.BatchNormalization()(x_4)
    
    x_1 = layers.ReLU()(x_1)
    x_2 = layers.ReLU()(x_2)
    x_3 = layers.ReLU()(x_3)
    x_4 = layers.ReLU()(x_4)
    
    x_1 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_1)
    x_2 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_2)
    x_3 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_3)
    x_4 = layers.Conv2D(filters=160, kernel_size=1, strides=1, padding='same')(x_4)
    
    x = tf.keras.layers.concatenate([x_1, x_2, x_3, x_4])
    
    x = layers.BatchNormalization()(x)
    
    x = layers.ReLU()(x)
    
    x = layers.AveragePooling2D((8,8))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(units=num_classes, activation='softmax')(x)
    
    return x