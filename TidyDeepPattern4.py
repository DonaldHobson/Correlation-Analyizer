import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import tensorflow as tf

#from scipy import misc
def normalize(x,axis=None):
    x=x-x.mean(axis=axis,keepdims=True)
    return x/((x**2).mean(axis=axis,keepdims=True))**.5

def _deepCorrelatee(data,output,dataTest,outputTest):
    data=data[:,:,None]
    dataTest=dataTest[:,:,None]
    channels=output.shape[1]
    a=Sequential()
    a.add(Conv1D(10,8,activation="relu",input_shape=data.shape[1:]))
    a.add(AveragePooling1D(pool_size=2))
    a.add(Conv1D(20,8,activation="relu"))
    a.add(AveragePooling1D(pool_size=2))
    a.add(Conv1D(30,8))
    a.add(Lambda(lambda x: tf.reduce_mean(x,axis=1)))
    a.add(Dense(50,activation="relu"))
    a.add(Dense(1))
    error=sigmoid_cross_entropy_with_logits(labels=outputTf,logits=a)

##    NNguesses=a.output[:,:channels]
##    NNlogErr=a.output[:,channels:]
##    NNerrors=tf.exp(NNlogErr)
##    outputTf=tf.placeholder(tf.float32, shape=(None,channels))
    #error=(tf.reduce_mean(NNerrors*(NNguesses-outputTf)**2)+tf.reduce_mean(1/NNerrors))*.5
    train=tf.train.AdamOptimizer(.004).minimize(error)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    x=np.zeros([53])#iterations
    

    def trainFunc():
        for i in np.nditer(x, op_flags=['readwrite']):
            shift=np.randint(1,data.shape[0])
            guesses=sess.run([error,train,NNlogErr],feed_dict={a.input:data,outputTf:output})
            guesses=sess.run([error,NNlogErr],feed_dict={a.input:dataTest,outputTf:outputTest})
            i[...]=guesses[0]
            yield guesses
            
    errorL=min(trainFunc(),key=lambda hhh:hhh[0])[1]
    #print(x[0],x[-1],x.min(),x.argmin())
    return (x.min(),errorL)

def deepCorrelate(data,output):
    data=normalize(data)
    output=normalize(output)
    samples=data.shape[0]
    assert samples==output.shape[0]
    v=np.random.permutation(samples)
    batches=10
    testing=2
    x=np.zeros(batches)
    NNerrors=np.zeros(output.shape)
    for j,i in  enumerate(np.nditer(x, op_flags=['readwrite'])):
        nr=(v+j)%batches<testing
        r=np.logical_not(nr)
        i[...] , newErrs =_deepCorrelatee(data[r],output[r],data[nr],output[nr])
        print(newErrs[:10])
        NNerrors[nr] += newErrs
        print(NNerrors.max(),NNerrors.mean(),"NNerrors")
    NNerrors/=testing
    print(x,x.mean())
    print(NNerrors.T,NNerrors.mean())
    return {"mean":errors.mean(),"all":errors}
if __name__=="__main__":
    from scipy import misc
    inpu=misc.imread("TestCorrelateData.png",flatten=True).T
    outpu=np.linspace(-1,1,150)[:,None]
    print(deepCorrelate(inpu,outpu))
