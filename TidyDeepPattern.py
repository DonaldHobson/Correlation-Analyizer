import numpy as np
from keras.layers import *
from keras.models import *
import tensorflow as tf

def normalize(x,axis=None):
    """returns x that has been linearly scaled so that x.mean(axis=axis)==zeros_like(x) and (x**2).mean(axis=axis)==ones_like(x)"""
    x=x-x.mean(axis=axis,keepdims=True)
    return x/((x**2).mean(axis=axis,keepdims=True))**.5

def _deepCorrelate(data,output,dataTest,outputTest):
    """Partial correlation code, private"""
    data=data[:,:,None]
    output=output[:,None,:]
    KerasSeq=Sequential()
    KerasSeq.add(Conv1D(10,8,activation="relu",input_shape=data.shape[1:]))
    KerasSeq.add(AveragePooling1D(pool_size=2))
    KerasSeq.add(Conv1D(20,8,activation="relu"))
    KerasSeq.add(AveragePooling1D(pool_size=2))
    KerasSeq.add(Conv1D(output.shape[2],8))
    outputTf=tf.placeholder(tf.float32, shape=(None,)+output.shape[1:])
    b=tf.shape(KerasSeq.output)
    norm=2/tf.reduce_prod(b)
    errorTf=tf.nn.l2_loss(KerasSeq.output-outputTf)*tf.to_float(norm)
    trainTf=tf.train.AdamOptimizer(.02).minimize(errorTf)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    errors=np.zeros(100)
    dataTest=dataTest[:,:,None]
    outputTest=outputTest[:,None,:]
    for error in np.nditer(errors, op_flags=['readwrite']):
        guesses=sess.run([trainTf],feed_dict={KerasSeq.input:data,outputTf:output})
        guesses=sess.run([errorTf],feed_dict={KerasSeq.input:dataTest,outputTf:outputTest})
        error[...]=guesses[0]
    return errors.min()

def deepCorrelate(data,output,repeats=10):
    """Correlates data and output. Both inputs dhould be numpy arrays of size (samples,timesteps) and (samples,channels) respectively.
    The output is a number between 0 and 1. 0 represents the maximum possible correlation and 1 means no correlation. Repeats is how many times to run algorithm.
    Higher values will be more acurate but slower."""
    data=normalize(data)
    output=normalize(output)
    errors=np.zeros(repeats)
    for error in  np.nditer(errors, op_flags=['readwrite']):
        #split into train and test samples randomly
        r=np.random.random(size=data.shape[0])<0.7
        nr=np.logical_not(r)
        error[...]=_deepCorrelate(data[r],output[r],data[nr],output[nr])
    return {"mean":errors.mean(),"all":errors}

if __name__=="__main__":
    from scipy import misc
    inpu=misc.imread("TestCorrelateData4.png",flatten=True).T
    outpu=np.linspace(-1,1,150)[:,None]
    print(deepCorrelate(inpu,outpu))
