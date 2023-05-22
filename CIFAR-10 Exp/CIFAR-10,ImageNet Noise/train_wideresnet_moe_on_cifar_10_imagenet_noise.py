

import numpy as np
import tensorflow as tf


from wideresnet_moe import WideResnet



for s in [20000]:
    
    #Loading the Data
    training_data_all = np.load('cifar_10_imagenet_noise_train_data.npy')
    training_label_all = np.load('cifar_10_imagenet_noise_train_label.npy')
    testing_data = np.load('cifar_10_imagenet_noise_test_data.npy')
    testing_label = np.load('cifar_10_imagenet_noise_test_label.npy')
    
    #sampling training data
    training_data=np.concatenate((training_data_all[0:0+(s//10)],training_data_all[5000:5000+(s//10)],training_data_all[10000:10000+(s//10)],training_data_all[15000:15000+(s//10)],training_data_all[20000:20000+(s//10)],training_data_all[25000:25000+(s//10)],training_data_all[30000:30000+(s//10)],training_data_all[35000:35000+(s//10)],training_data_all[40000:40000+(s//10)],training_data_all[45000:45000+(s//10)]),axis=0)
    
    training_label=np.concatenate((training_label_all[0:0+(s//10)],training_label_all[5000:5000+(s//10)],training_label_all[10000:10000+(s//10)],training_label_all[15000:15000+(s//10)],training_label_all[20000:20000+(s//10)],training_label_all[25000:25000+(s//10)],training_label_all[30000:30000+(s//10)],training_label_all[35000:35000+(s//10)],training_label_all[40000:40000+(s//10)],training_label_all[45000:45000+(s//10)]),axis=0)
    
    
    
    # 1-of-K encoding
    training_label = tf.reshape(tf.one_hot(training_label, axis=1, depth=10,dtype=tf.float64),(s,10)).numpy()
    testing_label = tf.reshape(tf.one_hot(testing_label, axis=1, depth=10, dtype=tf.float64),(10000,10)).numpy()
    
    
    
    #shuffling the training set
    indices = tf.range(start=0, limit=tf.shape(training_data)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    training_data = tf.gather(training_data, shuffled_indices,axis=0)
    training_label = tf.gather(training_label, shuffled_indices, axis=0)
    
    
    #Changing the data type to float32
    training_data=tf.cast(training_data,dtype=tf.dtypes.float32)
    testing_data=tf.cast(testing_data,dtype=tf.dtypes.float32)
    
    for i in range(5):
    
        #Creating the model
        model_input = tf.keras.Input(shape=(tf.shape(training_data)[1],tf.shape(training_data)[2],tf.shape(training_data)[3]))
        
        model_output = WideResnet(model_input, num_blocks=1, k=10, num_classes=10)
        
        
        #Model Aggregation
        model=tf.keras.Model(model_input,model_output)
        
        #Model Compilation
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss='categorical_crossentropy',
                      metrics='categorical_accuracy')
        
        #Call backs
        z=[]
        
        weights_dict = {}
        
        weight_callback = tf.keras.callbacks.LambdaCallback \
                                          ( on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:model.get_weights()}))
        
        
        testing_after_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: z.append(model.evaluate(testing_data, testing_label, batch_size=1000,verbose=1)))
        
        #Train the Model
        x=model.fit(training_data,training_label,batch_size=128,epochs=50,callbacks=[testing_after_epoch, weight_callback])
        
        f='test_acc_loss_cifar_10_imagenet_noise_wideresnet_moe_s_'+str(s//1000)+'k_v'+str(i+1)
        
        np.save(f,z)
        
        t = weights_dict[49]
        
        f='model_params_cifar_10_imagenet_noise_wideresnet_moe_s_'+str(s//1000)+'k_v'+str(i+1)
        
        np.save(f,t)



