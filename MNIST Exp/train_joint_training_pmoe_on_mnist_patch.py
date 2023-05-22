

import numpy as np
import tensorflow as tf


from gater import gate


for s in [100, 300, 500, 700, 900, 1000]:
    for l in [6]:
        #Loading Data
        training_data_all = np.load('Training_Data_patch_mnist_n_16.npy')
        training_label_all = np.load('Training_Label_patch_mnist_n_16.npy')
        testing_data = np.load('Test_Data_patch_mnist_n_16.npy')
        testing_label = np.load('Test_Label_patch_mnist_n_16.npy')
        testing_p_pos=np.load('Test_pattern_pos_mnist_n_16.npy')
        
        #Sampling required no. of training samples
        training_data=np.concatenate((training_data_all[0:0+(s//2)],training_data_all[5000:5000+(s//2)]),axis=0)
        training_label=np.concatenate((training_label_all[0:0+(s//2)],training_label_all[5000:5000+(s//2)]),axis=0)
        
        
        
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
        
        
        #Initializers
        initializer_gate=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.0001)
        initializer_cnnl=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.01)
        initializer_fcl=tf.keras.initializers.RandomNormal(mean=0.0,stddev=1)
        
        for i in range(5):
            #Creating the model
            
            #Input Layer
            model_input=tf.keras.Input(shape=(tf.shape(training_data)[1],tf.shape(training_data)[2],tf.shape(training_data)[3]))
            
            
            
            #Conv Layer
            
            x_1 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            x_2 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            x_3 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            x_4 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            x_5 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            x_6 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            x_7 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            x_8 = gate(l,(28,28),(28,28),gating_activation=tf.nn.softmax,gating_kernel_initializer=initializer_gate)(model_input)
            
            
            x_1 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_1)
            x_2 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_2)
            x_3 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_3)
            x_4 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_4)
            x_5 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_5)
            x_6 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_6)
            x_7 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_7)
            x_8 = tf.keras.layers.Conv2D(5, (28,28), strides=28, padding='valid', data_format='channels_last', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(x_8)
            
            x_5=tf.keras.layers.Concatenate()([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8])
            
            #Pooling Layer
            x_5=tf.keras.layers.GlobalAveragePooling2D()(x_5)
            
            #Output Layer
            model_output=tf.keras.layers.Dense(1,activation='linear',use_bias=False, trainable=False, kernel_initializer=initializer_fcl, name='FCL')(x_5)
            
            #Model Aggregation
            model=tf.keras.Model(model_input,model_output)
            
            #Model Compilation
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=tf.keras.metrics.BinaryAccuracy(),run_eagerly=True)
            
            #Call backs
            z=[]
            
            weights_dict = {}
            
            weight_callback = tf.keras.callbacks.LambdaCallback \
                                              ( on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:model.get_weights()}))
            
            
            testing_after_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: z.append(model.evaluate(testing_data, testing_label, batch_size=1000,verbose=1)))
            
            #Train the Model
            x=model.fit(training_data,training_label,batch_size=20,epochs=150,callbacks=[testing_after_epoch, weight_callback])
            
            
            f='test_acc_loss_mnist_joint_train_router_s_'+str(s)+'_v'+str(i+1)
                    
            np.save(f,z)
            
           


