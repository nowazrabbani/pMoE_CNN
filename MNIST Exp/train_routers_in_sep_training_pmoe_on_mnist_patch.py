
import numpy as np
import tensorflow as tf


def linear_loss (y,f):
    y=tf.cast(y, dtype=tf.dtypes.float32)
    linear_loss=-(y*f)
    return linear_loss

def custom_acc(y,f):
    total=len(y)
    y=tf.cast(y, dtype=tf.dtypes.float32)
    f_1=y*f
    count=len(tf.where(f_1>0))
    return count/total

for s in [100, 300, 500, 700, 900, 1000]:
    #Loading Data
    training_data_all = np.load('Training_Data_patch_mnist_n_16.npy')
    training_label_all = np.load('Training_Label_patch_mnist_n_16.npy')
    testing_data = np.load('Test_Data_patch_mnist_n_16.npy')
    testing_label = np.load('Test_Label_patch_mnist_n_16.npy')
    
    #Sampling required no. of samples
    training_data=np.concatenate((training_data_all[0:0+(s//2)],training_data_all[5000:5000+(s//2)]),axis=0)
    training_label=np.concatenate((training_label_all[0:0+(s//2)],training_label_all[5000:5000+(s//2)]),axis=0)
    
    
    #Redefining labels
    training_label[0:0+(s//2)]=training_label[0:0+(s//2)]-1
    testing_label[0:499]=testing_label[0:499]-1
    
    #shuffling the training set
    indices = tf.range(start=0, limit=tf.shape(training_data)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    training_data = tf.gather(training_data, shuffled_indices,axis=0)
    training_label = tf.gather(training_label, shuffled_indices, axis=0)
    
    
    #normalizing and reshaping data
    training_data=training_data/255
    testing_data=testing_data/255
    training_data=tf.reshape(training_data,(tf.shape(training_data)[0],tf.shape(training_data)[1],tf.shape(training_data)[2],1))
    testing_data=tf.reshape(testing_data,(tf.shape(testing_data)[0],tf.shape(testing_data)[1],tf.shape(testing_data)[2],1))
    
 
    #Creating the model
    model=tf.keras.Sequential()
    initializer_cnnl=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1)
    initializer_fcl=tf.keras.initializers.Constant(value=[16,-16])
    model.add(tf.keras.layers.Conv2D(2, (28,28), strides=28, padding='valid', activation='linear', use_bias=False, kernel_initializer=initializer_cnnl, name='CNNL', input_shape=(112,112,1)))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(1,activation='linear',use_bias=False, trainable=False, kernel_initializer=initializer_fcl, name='FCL'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),loss=linear_loss,metrics=custom_acc)
    
    z=[]
    
    #Call backs
    testing_after_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: z.append(model.evaluate(testing_data, testing_label, batch_size=1000,verbose=0)))
    
    #Train the Model
    x=model.fit(training_data,training_label,batch_size=20,epochs=100,callbacks=testing_after_epoch)
    
    f='Router_Weights_16_s_'+str(s)
    t=model.layers[0].get_weights()
    t=t[0]
    np.save(f,t)
