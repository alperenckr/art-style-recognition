import layers
import numpy as np
import CNN
from solver import sgd, get_minibatch
import h5py
import preprocessing as pre


def reshape(mat,size,batch=16):
    new_mat = np.ones((mat.shape[0],size,size))
    for i in range(mat.shape[0]):
        new_mat[i] = np.ones((size,size))*mat[i]
    new_mat = np.tile(new_mat,(batch,1,1,1))
    return new_mat


def get_from_file(file_path="low_loss.hdf5"):
    f = h5py.File(file_path,"r")
    datasetNames = [n for n in f['model_weights'].keys()]
    print(datasetNames)
    print(f['model_weights']['norm1']['norm1_2'].keys())

    weights = {}
    weights['conv1_kernel'] = f['model_weights']['conv1']['conv1_2']['kernel:0'].value.transpose(3,2,1,0)
    weights['conv1_bias']   = f['model_weights']['conv1']['conv1_2']['bias:0'].value
    weights['conv2_kernel'] = f['model_weights']['conv2']['conv2_2']['kernel:0'].value.transpose(3,2,1,0)
    weights['conv2_bias']   = f['model_weights']['conv2']['conv2_2']['bias:0'].value
    weights['conv3_kernel'] = f['model_weights']['conv3']['conv3_2']['kernel:0'].value.transpose(3,2,1,0)
    weights['conv3_bias']   = f['model_weights']['conv3']['conv3_2']['bias:0'].value
    weights['conv4_kernel'] = f['model_weights']['conv4']['conv4_2']['kernel:0'].value.transpose(3,2,1,0)
    weights['conv4_bias']   = f['model_weights']['conv4']['conv4_2']['bias:0'].value
    weights['conv5_kernel'] = f['model_weights']['conv5']['conv5_2']['kernel:0'].value.transpose(3,2,1,0)
    weights['conv5_bias']   = f['model_weights']['conv5']['conv5_2']['bias:0'].value
    weights['fc6_kernel']   = f['model_weights']['fc6']['fc6_2']['kernel:0'].value
    weights['fc6_bias']     = f['model_weights']['fc6']['fc6_2']['bias:0'].value
    weights['fc7_kernel']   = f['model_weights']['fc7']['fc7_2']['kernel:0'].value
    weights['fc7_bias']     = f['model_weights']['fc7']['fc7_2']['bias:0'].value
    weights['fc8_kernel']   = f['model_weights']['fc8a']['fc8a_2']['kernel:0'].value
    weights['fc8_bias']     = f['model_weights']['fc8a']['fc8a_2']['bias:0'].value
    weights['norm1_beta']   = reshape(f['model_weights']['norm1']['norm1_2']['beta:0'].value,27)
    weights['norm1_gamma']  = reshape(f['model_weights']['norm1']['norm1_2']['gamma:0'].value,27)
    weights['norm1_mean']   = reshape(f['model_weights']['norm1']['norm1_2']['moving_mean:0'].value,27,1)
    weights['norm1_var']    = reshape(f['model_weights']['norm1']['norm1_2']['moving_variance:0'].value,27,1)
    weights['norm2_beta']   = reshape(f['model_weights']['norm2']['norm2_2']['beta:0'].value,13)
    weights['norm2_gamma']  = reshape(f['model_weights']['norm2']['norm2_2']['gamma:0'].value,13)
    weights['norm2_mean']   = reshape(f['model_weights']['norm2']['norm2_2']['moving_mean:0'].value,13,1)
    weights['norm2_var']    = reshape(f['model_weights']['norm2']['norm2_2']['moving_variance:0'].value,13,1)


    return weights

#weights = get_from_file()
X_train, Y_train, X_test, Y_test = pre.get_pictures_from_zip("train_prep_latest.zip",pre.train_df)
minibatch = get_minibatch(np.array(X_test),np.array(Y_test),16)
#print(len(X_test))
#print(len(Y_test))
#print(X_test[0].shape)
nn = CNN.OurConvNet()
#Pretrained weights
a = sgd(nn, X, y, None, 0.001, 16, 10, 1) #Egitim icin
for bat in minibatch:
    nn.prediction(bat[0],bat[1])
    print("==============")

