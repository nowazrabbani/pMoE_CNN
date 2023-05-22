
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def avg_stddev_calc(f,no_v):
    
    f_t=f+'_v'+str(1)+'.npy'
    t_1 = np.load(f_t)
    t_1=tf.reshape(t_1,(1,tf.shape(t_1)[0],tf.shape(t_1)[1])).numpy()
    for i in range(1,no_v):
        f_t=f+'_v'+str(i+1)+'.npy'
        t=np.load(f_t)
        t=tf.reshape(t,(1,tf.shape(t)[0],tf.shape(t)[1])).numpy()
        t_1=tf.concat((t_1,t),axis=0).numpy()
    
    t_av=tf.math.reduce_mean(t_1,axis=0).numpy()
    t_std=tf.math.reduce_std(t_1,axis=0).numpy()
    
    return t_av, t_std
    

def last_epoch_result_collection(f,no_sample_points, points, no_v=5, last_epoch=50):
    
    t_av_s=np.zeros((no_sample_points,3),dtype=np.float64)
    t_std_s=np.zeros((no_sample_points,3),dtype=np.float64)
    
    for i in range(no_sample_points):
        f_1=f+'_s_'+str(points[i])
        t_av, t_std =avg_stddev_calc(f_1,no_v)
        
        t_av_s[i,0]=t_av[last_epoch-1,0]
        t_av_s[i,1]=t_av[last_epoch-1,1]
        t_av_s[i,2]=points[i]
        
        t_std_s[i,0]=t_std[last_epoch-1,0]
        t_std_s[i,1]=t_std[last_epoch-1,1]
        t_std_s[i,2]=points[i]
        
    return t_av_s, t_std_s


f='test_acc_loss_mnist_CNN'
f_moe='test_acc_loss_mnist_MoE_hard'
f_moe_j='test_acc_loss_mnist_joint_train_router'
no_sample_points=6
points=[100, 300, 500, 700, 900, 1000]
no_v=5
last_epoch=150


wideresnet_av_s, wideresnet_std_s = last_epoch_result_collection(f, no_sample_points, points, no_v, last_epoch)
wideresnet_moe_av_s, wideresnet_moe_std_s = last_epoch_result_collection(f_moe, no_sample_points, points, no_v, last_epoch)
wideresnet_moe_j_av_s, wideresnet_moe_j_std_s = last_epoch_result_collection(f_moe_j, no_sample_points, points, no_v, last_epoch)
wideresnet_av_s=wideresnet_av_s*100
wideresnet_std_s=wideresnet_std_s*100
wideresnet_moe_av_s=wideresnet_moe_av_s*100
wideresnet_moe_std_s=wideresnet_moe_std_s*100
wideresnet_moe_j_av_s=wideresnet_moe_j_av_s*100
wideresnet_moe_j_std_s=wideresnet_moe_j_std_s*100

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 60
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = [16, 12]

plt.errorbar(points,wideresnet_av_s[:,1],wideresnet_std_s[:,1],marker='d', markersize=30,label='CNN',lw=8)
plt.errorbar(points,wideresnet_moe_av_s[:,1],wideresnet_moe_std_s[:,1],marker='o',markersize=30,label='pMoE(separate training)',lw=8)
plt.errorbar(points,wideresnet_moe_j_av_s[:,1],wideresnet_moe_j_std_s[:,1],marker='^',markersize=30,label='pMoE(joint training)',lw=8)
plt.axhline(y=wideresnet_av_s[no_sample_points-1,1],c='k',ls=':',lw=4)
plt.axvline(x=600,c='k',ls=':',lw=4)
plt.axvline(x=1000,c='k',ls=':',lw=4)
plt.legend(loc='lower right',ncol=1,prop={'size': 50},frameon=False)
plt.xlabel('No. of training samples ($N$)',fontweight='bold')
plt.ylabel('Test accuracy(\%)',fontweight='bold')
plt.show()

# =============================================================================
# plt.errorbar(wideresnet_av_s[:,2],wideresnet_av_s[:,0],wideresnet_std_s[:,0],marker='d',markersize=30,label='CNN')
# plt.errorbar(wideresnet_moe_av_s[:,2],wideresnet_moe_av_s[:,0],wideresnet_moe_std_s[:,0],marker='v',markersize=30,label='pMoE(fixed router)')
# plt.errorbar(wideresnet_moe_j_av_s[:,2],wideresnet_moe_j_av_s[:,0],wideresnet_moe_j_std_s[:,0],marker='s',markersize=30,label='pMoE(joint training)')
# plt.legend(loc='lower right',ncol=1,prop={'size': 40},frameon=False)
# plt.xlabel('No. of training samples ($N$)',fontweight='bold')
# plt.ylabel('Test loss',fontweight='bold')
# plt.show()
# =============================================================================

        