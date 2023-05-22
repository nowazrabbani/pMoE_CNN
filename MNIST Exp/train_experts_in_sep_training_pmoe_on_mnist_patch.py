

import numpy as np
import tensorflow as tf


for s in [100, 300, 500, 700, 900, 1000]:
    
    for l in [2]:
    
        #Loading Data
        training_data_all = np.load('Training_Data_patch_mnist_n_16.npy')
        training_label_all = np.load('Training_Label_patch_mnist_n_16.npy')
        testing_data = np.load('Test_Data_patch_mnist_n_16.npy')
        testing_label = np.load('Test_Label_patch_mnist_n_16.npy')
        
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
        
        f='Router_Weights_16_s_'+str(s)+'.npy'
        #Loading Router
        router_weights=np.load(f)
        
        #Sampling patches
        def router_outputs(r,d,l):
            d=tf.cast(d,dtype=tf.dtypes.float32)
            d=tf.reshape(d,(tf.shape(d)[0],tf.shape(d)[1],tf.shape(d)[2],tf.shape(r)[2]))
            t=tf.nn.conv2d(d,r,strides=len(r),padding='VALID')
            t_1=tf.reshape(tf.transpose(t,perm=(0,3,2,1)),(tf.shape(t)[0],tf.shape(t)[3],
                                                       tf.shape(t)[1]*tf.shape(t)[2]))
            [t_2,t_3]=tf.math.top_k(t_1,l)
            
            t_1=t_1.numpy()
            
            x=tf.shape(t_1)[0].numpy()
            y=tf.shape(t_1)[1].numpy()
            z=tf.shape(t_1)[2].numpy()
            
            for i in range(x):
                for j in range(y):
                    for k in range(z):
                        if tf.math.reduce_any(t_3[i,j,:]==k):
                            t_1[i,j,k]=1
                        else:
                            t_1[i,j,k]=0
        
            t_4=tf.cast(tf.shape(t_1)[2],dtype=tf.dtypes.float32)
            t_4=tf.cast(tf.math.sqrt(t_4),dtype=tf.dtypes.int32)
            t_1=tf.reshape(t_1,(tf.shape(t_1)[0],tf.shape(t_1)[1],t_4,t_4))
            t_1=tf.transpose(t_1,perm=(0,3,2,1))
            t_1=tf.repeat(t_1,len(r)*len(r),axis=3)
            d=tf.image.extract_patches(d, [1,len(r),len(r),1], [1,len(r),len(r),1], [1,1,1,1], padding='VALID')
            d=d*t_1
            d=tf.reshape(d,(tf.shape(d)[0],tf.shape(d)[1],tf.shape(d)[2],len(r),len(r),tf.shape(r)[2]))
            d=tf.transpose(d,perm=(0,1,3,2,4,5))
            d=tf.reshape(d,(tf.shape(d)[0],tf.shape(d)[1]*tf.shape(d)[2],tf.shape(d)[3]*tf.shape(d)[4],tf.shape(d)[5]))
            d=tf.cast(d,dtype=tf.dtypes.float32)
            
            return d
        
        r_1=tf.reshape(router_weights[:,:,:,0],(tf.shape(router_weights)[0],tf.shape(router_weights)[1],tf.shape(router_weights)[2],1))
        r_2=tf.reshape(router_weights[:,:,:,1],(tf.shape(router_weights)[0],tf.shape(router_weights)[1],tf.shape(router_weights)[2],1))
        training_data_1=router_outputs(r_1,training_data,l)
        training_data_1=tf.reshape(training_data_1,(tf.shape(training_data_1)[0],tf.shape(training_data_1)[1],tf.shape(training_data_1)[2],tf.shape(training_data_1)[3],1))
        training_data_2=router_outputs(r_2,training_data,l)
        training_data_2=tf.reshape(training_data_2,(tf.shape(training_data_2)[0],tf.shape(training_data_2)[1],tf.shape(training_data_2)[2],tf.shape(training_data_2)[3],1))
        training_data=tf.concat((training_data_1,training_data_2),axis=4)
        testing_data_1=router_outputs(r_1,testing_data,l)
        testing_data_1=tf.reshape(testing_data_1,(tf.shape(testing_data_1)[0],tf.shape(testing_data_1)[1],tf.shape(testing_data_1)[2],tf.shape(testing_data_1)[3],1))
        testing_data_2=router_outputs(r_2,testing_data,l)
        testing_data_2=tf.reshape(testing_data_2,(tf.shape(testing_data_2)[0],tf.shape(testing_data_2)[1],tf.shape(testing_data_2)[2],tf.shape(testing_data_2)[3],1))
        testing_data=tf.concat((testing_data_1,testing_data_2),axis=4)
        
        for i in range(5):
            #Initializers
            initializer_cnnl=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1)
            #initializer_fcl=tf.keras.initializers.RandomNormal(mean=0.0,stddev=1)
            val_1=np.ones((20,))
            val_2=-1*val_1
            val=np.concatenate((val_1,val_2))
            initializer_fcl=tf.keras.initializers.Constant(value=val)
            
            #Creating the model
            
            model_input=tf.keras.Input(shape=(tf.shape(training_data)[1],tf.shape(training_data)[2],tf.shape(training_data)[3],tf.shape(training_data)[4]))
            x_1=tf.keras.layers.Conv2D(20, (28,28), strides=28, padding='valid', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(model_input[:,:,:,:,0])
            x_2=tf.keras.layers.Conv2D(20, (28,28), strides=28, padding='valid', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl)(model_input[:,:,:,:,1])
            x_3=tf.keras.layers.Concatenate()([x_1, x_2])
            x_3=tf.keras.layers.GlobalAveragePooling2D()(x_3)
            model_output=tf.keras.layers.Dense(1,activation='linear',use_bias=False, trainable=False, kernel_initializer=initializer_fcl, name='FCL')(x_3)
            
            model=tf.keras.Model(model_input,model_output)
            
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=tf.keras.metrics.BinaryAccuracy(),run_eagerly=True)
            
            z=[]
            
            #Call backs
            testing_after_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: z.append(model.evaluate(testing_data, testing_label, batch_size=1000,verbose=1)))
            
            #Train the Model
            x=model.fit(training_data,training_label,batch_size=20,epochs=150,callbacks=testing_after_epoch)
            
            f='test_acc_loss_mnist_MoE_hard_s_'+str(s)+'_v'+str(i+1)
                    
            np.save(f,z)

