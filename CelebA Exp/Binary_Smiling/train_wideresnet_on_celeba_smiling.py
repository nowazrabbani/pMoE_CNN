

import numpy as np
import tensorflow as tf


from wideresnet_bin import WideResnet



for s in [2000, 4000, 6000, 8000, 10000]:
    
    #Loading the Data
    training_data_all = np.load('train_data_celeba_smiling_64.npy')
    training_label_all = np.load('train_label_celeba_smiling.npy')
    testing_data = np.load('test_data_celeba_smiling_64.npy')
    testing_label = np.load('test_label_celeba_smiling.npy')
    
    #sampling training data
    training_data=np.concatenate((training_data_all[0:0+(s//4)],training_data_all[25000:25000+(s//4)],training_data_all[50000:50000+(s//4)],training_data_all[75000:75000+(s//4)]),axis=0)
    
    training_label=np.concatenate((training_label_all[0:0+(s//4)],training_label_all[25000:25000+(s//4)],training_label_all[50000:50000+(s//4)],training_label_all[75000:75000+(s//4)]),axis=0)
    
    
    #shuffling the training set
    indices = tf.range(start=0, limit=tf.shape(training_data)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    training_data = tf.gather(training_data, shuffled_indices,axis=0)
    training_label = tf.gather(training_label, shuffled_indices, axis=0)
    
    
    for i in range(5):
    
        #Creating the model
        model_input = tf.keras.Input(shape=(tf.shape(training_data)[1],tf.shape(training_data)[2],tf.shape(training_data)[3]))
        
        model_output = WideResnet(model_input, num_blocks=1, k=10, num_classes=1)
        
        
        #Model Aggregation
        model=tf.keras.Model(model_input,model_output)
        
        #Model Compilation
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=tf.keras.metrics.BinaryAccuracy())
        
        #Call backs
        z=[]
        
        weights_dict = {}
        
        weight_callback = tf.keras.callbacks.LambdaCallback \
                                          ( on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:model.get_weights()}))
        
        
        testing_after_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: z.append(model.evaluate(testing_data, testing_label, batch_size=100,verbose=1)))
        
        #Train the Model
        x=model.fit(training_data,training_label,batch_size=128,epochs=50,callbacks=[testing_after_epoch, weight_callback])
        
        f='test_acc_loss_celeba_smiling_wideresnet_s_'+str(s//1000)+'k_v'+str(i+1)
        
        np.save(f,z)
        
        t = weights_dict[49]
        
        f='model_params_celeba_smiling_wideresnet_s_'+str(s//1000)+'k_v'+str(i+1)
        
        np.save(f,t)



