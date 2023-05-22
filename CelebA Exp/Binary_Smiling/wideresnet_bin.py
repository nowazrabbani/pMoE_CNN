

from tensorflow.keras import layers



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
    
    x = WideResnetGroup(x, num_blocks, widths[2], strides=(2, 2))
    
    x = layers.BatchNormalization()(x)
    
    x = layers.ReLU()(x)
    
    x = layers.AveragePooling2D((16,16))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(units=num_classes, activation='linear')(x)
    
    return x