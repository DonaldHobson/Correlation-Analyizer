import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import tensorflow as tf

from scipy import misc
import pdb
#pdb.pm()
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def show(x):
    plt.imshow(x.reshape((28,28)), cmap="Greys")
    plt.show()
#relu 0.83 acc    
#tanh 0.89
#assert 0
def normalize(x,axis=None):
    x=x-x.mean(axis=axis,keepdims=True)
    return x/((x**2).mean(axis=axis,keepdims=True))**.5
def normPlot(*x):
    for i in x:
        plt.plot(normalize(i.ravel()))
    plt.show()
#gmin=np.random.normal(size=(1,100,1))*0.01
#gmax=np.random.normal(size=(1,100,1))*0.01
def _deepCorrelatee(data,output,dataTest,outputTest):
   # a,b,at,bt=data,output,dataTest,outputTest
    #pdb.pm()
    #assert 0
    data=data[:,:,None]
    dataTest=dataTest[:,:,None]
    channels=output.shape[1]
    a=Sequential()
    #a.add(Reshape((28,28,1),input_shape=(784,)))
    a.add(Conv1D(10,8,activation="relu",input_shape=data.shape[1:]))
    a.add(AveragePooling1D(pool_size=2))
    a.add(Conv1D(20,8,activation="relu"))
    a.add(AveragePooling1D(pool_size=2))
    a.add(Conv1D(30,8))
    #print(a.output_shape)
    a.add(Lambda(lambda x: tf.reduce_mean(x,axis=1)))
    #print(a.output_shape)
    a.add(Dense(50,activation="relu"))
    #a.add(Dense(50,activation="relu"))
    a.add(Dense(channels*2))
    
    #channels*2
    #NNout=
    NNguesses=a.output[:,:channels]
    NNlogErr=a.output[:,channels:]
    NNerrors=tf.exp(NNlogErr)
    
    outputTf=tf.placeholder(tf.float32, shape=(None,channels))
    
    #dataTf=tf.placeholder(tf.float32, shape=data.shape)
    
    #b=tf.shape(a.output)
    #print(b)
    #norm=1/tf.reduce_prod(b)
    error=(tf.reduce_mean(NNerrors*(NNguesses-outputTf)**2)+tf.reduce_mean(1/NNerrors))*.5#*tf.to_float(norm)
    
    
    #error=tf.nn.l2_loss(a.output-outputTf)*tf.to_float(norm)
    
    train=tf.train.AdamOptimizer(.004).minimize(error)
    #print(a.output.shape,error)
    #print(a,a.output)
    #assert 0
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    x=np.zeros([53])#iterations
    y=np.zeros_like(outputTest)


    #dataTest=normalize(dataTest)[:,:,None]
    #outputTest=normalize(outputTest)[:,:]
    
    #print(
    #ainp=dataTest#np.random.normal(size=dataTest.shape)
    #aoup=outputTest#np.random.normal(size=outputTest.shape)
    def trainFunc():
        
        for i in np.nditer(x, op_flags=['readwrite']):
            
            guesses=sess.run([error,train,NNlogErr],feed_dict={a.input:data,outputTf:output})
            #if i==0:
            #    print("\t"*3,guesses[0])
            #print(guesses[0],"train")#,guesses[2][:10,0])
            guesses=sess.run([error,NNlogErr],feed_dict={a.input:dataTest,outputTf:outputTest})
            #guesses=sess.run([error,NNlogErr],feed_dict={a.input:ainp,outputTf:aoup})
            #print(guesses[0])#,guesses[1][:10,0])
                
            i[...]=guesses[0]
            yield guesses
    #trainFunc()      
    errorL=min(trainFunc(),key=lambda hhh:hhh[0])[1]
    #guesses=sess.run([error,NNlogErr],feed_dict={a.input:dataTest,outputTf:outputTest})
    
    #print(x)
    print(x[0],x[-1],x.min(),x.argmin())
    #error_=sess.run([NNlogErr],feed_dict={a.input:dataTest,outputTf:outputTest})
    
    #print()
    #print(guesses[0])
    #h=tf.reduce_mean(a.output)
    #grad=tf.gradients(h,a.input)[0]
    #global gmin,gmax
    #for i in range(2):
    #    gmin+=sess.run([grad],feed_dict={a.input:gmin})[0]
    #    gmax-=sess.run([grad],feed_dict={a.input:gmax})[0]
    return (x.min(),errorL)
def deepCorrelate(data,output):
    data=normalize(data)
    output=normalize(output)
    samples=data.shape[0]
    assert samples==output.shape[0]
    v=np.random.permutation(samples)
    batches=10
    testing=2
    
    
    x=np.zeros(batches)#10
    NNerrors=np.zeros(output.shape)
    for j,i in  enumerate(np.nditer(x, op_flags=['readwrite'])):
        nr=(v+j)%batches<testing
        #r=np.random.random(size=data.shape[0])<0.7
        r=np.logical_not(nr)
        i[...] , newErrs =_deepCorrelatee(data[r],output[r],data[nr],output[nr])
        print(newErrs[:10])
        NNerrors[nr] += newErrs
        print(NNerrors.max(),NNerrors.mean(),"NNerrors")
    NNerrors/=testing
    print(x,x.mean())
    print(NNerrors.T,NNerrors.mean())
#assert 0
    
inpu=misc.imread("TestCorrelateData.png",flatten=True).T
    
outpu=np.linspace(-1,1,150)[:,None]
deepCorrelate(inpu,outpu)
#assert 0
#inpu=np.random.normal(size=(150,100))
#outpu=np.random.normal(size=(150,3))
#inpu+=outpu[:,0,None]
#_deepCorrelatee(inpu[:100],outpu[:100],inpu[100:],outpu[100:])
#deepCorrelate(inpu,outpu)
#print((outpu**2).sum()*6)
assert 0
