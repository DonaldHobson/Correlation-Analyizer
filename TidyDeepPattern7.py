import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
import tensorflow as tf
import pdb
from scipy.special import expit as sigmoid

def normalize(x,axis=None):
    """returns x that has been linearly scaled so that
    x.mean(axis=axis)==zeros_like(x) and
    (x**2).mean(axis=axis)==ones_like(x)
    """
    xmean=x.mean(axis=axis,keepdims=True)
    xstd=x.std(axis=axis,keepdims=True)
    
    def f(y):
        return (y-xmean)/xstd
    
    return f
#def sigmoid(x):
#	return np.sign(x)*(1/(1+np.exp(-np.absolute(x)))-.5)+.5
def invSig(x):
    return np.log(1/(1-x)-1)
def zerOne(x):
    """Returns a numpy array containing x zeros followed by x ones."""
    y=np.zeros([x*2,1],np.float32)
    y[x:]=1
    return y


def preprocess(data,cycle):
    """When called, returns cycled data followed by plain data.
    Cycling is done so that the last item is moved to the front.
    If cycle is false then neither part of the data is cycled and
    this function just returns two copies of the data.
    """
    s=data.shape[0]
    i=0
    if cycle:
        while True:
            i+=1
            if i==0: continue   #to ensure the data is permuted
            i=i%s
            p=np.arange(i,i+s)%s
            #p is a permutation of np.arange used to reorder data.
            
            yield np.concatenate([data[p],data],axis=0)
    else:
        doubled=np.concatenate([data,data],axis=0)
        while True:
            yield doubled


def makeModel(dataShape,layers):
    """ returns a keras sequential model. This must be done repeatedly
    because their is no way to reset the model to random starting values
    """
    assert len(dataShape)==2
    while True:
        model=Sequential()
        model.add(Dense(layers[0],input_shape=dataShape[1:],activation="relu"))
        for i in layers[1:]:
            model.add(Dense(i,activation="relu"))
        yield model
def makeConvModel(dataShape,convFilters, convKernal, layers,convDim=1):
    """ returns a keras sequential model. This must be done repeatedly
    because their is no way to reset the model to random starting values
    """
    if convDim==1:
        ConvLayer=Conv1D
    elif convDim==2:
        ConvLayer=Conv2D
    else:
        raise BaseException("ConvDim must be 1 or 2, for Conv1D and Conv2D filters respectively.")
    
    assert len(dataShape)==3
    while True:
        model=Sequential()
        model.add(ConvLayer(convFilters[0],convKernal[0],input_shape=dataShape[1:],activation="relu"))
        for i in zip(convFilters[1:],convKernal[1:]):
            model.add(ConvLayer(*i,activation="relu"))
        model.add(Lambda(lambda x: tf.reduce_mean(x,axis=1)))
        for i in layers:
            model.add(Dense(i,activation="relu"))
        yield model
        

##class DSize():
##    """Small module to hold onto any parameters that can't change between different
##    pieces of dato being correlated. like size of data and how it is split.
##    """
##    
##    def __init__(self,tr=.6):
##        self.tr=tr
##        self.lenTrain=None
##        self.lenTest=None
##        self.allOut=0
##        
##    def _addDataShape(self,dataShape,lastlayer):
##        self.allOut+=lastlayer
##        if self.lenTrain==self.lenTest==None:
##            self.lenTrain=int(dataShape[0]*self.tr)
##            self.lenTest=dataShape[0]-self.lenTrain
##        else:
##            assert self.lenTrain==int(dataShape[0]*self.tr),(self.lenTrain,int(dataShape[0]*self.tr))
##            
##    def data(self,*args,**kwargs):
##        """data: a numpy ndarray of data, layers =[5,5] describes depth
##        and bredth of neural net ,second:controls what correlates to
##        what.
##        """
##        return Data(self,*args,**kwargs)
##    def convData(self,*args,**kwargs):
##        """data: a numpy ndarray of data, layers =[5,5] describes depth
##        and bredth of neural net ,second:controls what correlates to
##        what.
##        """
##        return ConvData(self,*args,**kwargs)

        
class Data():
    perm=None
    
    def __init__(self,data,split=None,layers=[5,5],second=False,normal=False):
        
        if split==None:
            split=data.shape[0]-1
        if split<1:
            split=int(split*data.shape[0])
        self.lenTrain=split
        self.lenTest=data.shape[0]-split
        if data.ndim==1:
            data=data[:,None]
        if Data.perm is None:
            Data.perm=np.random.permutation(data.shape[0])
        data=data[Data.perm]
        
        if not normal:
            self.normalizer=normalize(data,axis=0)
            data=self.normalizer(data)
        else:
            self.normalizer=lambda x:x

        
        self.train=preprocess(data[:split],second)
        self.test=preprocess(data[split:],second)
        self.shape=data.shape
        
        
        self.model=makeModel(data.shape,layers=layers)
class ConvData(Data):
    
    def __init__(self,dsize,data,convFilters, convKernal, layers=[],convDim=1,second=False):
        assert len(convFilters)==len(convKernal)
        self.dsize=dsize
        self.dsize._addDataShape(data.shape,(layers or convFilters)[-1])
        self.lenTrain=self.dsize.lenTrain
        self.lenTest=self.dsize.lenTest
        while data.ndim<3:
            data=data[...,None]
        if Data.perm is None:
            Data.perm=np.random.permutation(data.shape[0])
        data=data[Data.perm]
        self.normalizer=normalize(data)
        data=self.normalizer(data)
        split=self.dsize.lenTrain
        
        self.train=preprocess(data[:split],second)
        self.test=preprocess(data[split:],second)
        self.shape=data.shape
        
        
        self.model=makeConvModel(data.shape,convFilters, convKernal, layers,convDim)

def correlate(*data,repeats=3,iterations=50,returnFunc=False,train=.6):
    """Correlates data and output. All inputs should be 2D numpy arrays
    with the same size[0]. The output is a number between 0 and 1. 0
    represents the maximum possible correlation and 1 means no
    correlation. Repeats is how many times to run algorithm. Higher
    values will be more accurate but slower.

    The algorithm actually measures the amount of information needed to
    tell whether a pair of values are matched or not based on the
    probabilities from the neural network.
    """
    trainSeq=zerOne(data[0].lenTrain)
    testSeq=zerOne(data[0].lenTest)
    errorsNp=np.zeros([repeats,iterations])
    funcs=[]
    for thisErrors in errorsNp:#np.nditer(errorsNp, op_flags=['readwrite']):#www in range(repeats):
        print(thisErrors.shape)
        oldModels=[next(i.model) for i in data]
        oldOutputs=[i.output for i in oldModels]
        oldInputs=[i.input for i in oldModels]
        colapsed=tf.concat(oldOutputs,1)
        model=Sequential()
        model.add(Dense(20,input_shape=[int(colapsed.shape[1])],activation="relu"))
        model.add(Dense(20,activation="relu"))
        model.add(Dense(1))
        outputTf=tf.placeholder(tf.float32, shape=(None,1))
        gu=model(colapsed)
        error=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=outputTf,logits=gu))/np.log(2)
        train=tf.train.AdamOptimizer(.04).minimize(error)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        x=np.zeros([iterations])#iterations
        
        def trainFunc():
            for i in np.nditer(thisErrors, op_flags=['readwrite']):
                feed_dict={p:next(q.train) for p,q in zip(oldInputs,data)}
                feed_dict[outputTf]=trainSeq
                guesses=sess.run([error,train],feed_dict=feed_dict)
                #print(guesses,"train")
                feed_dict={p:next(q.test) for p,q in zip(oldInputs,data)}
                feed_dict[outputTf]=testSeq
                #print(list(i.shape for i in feed_dict.values()))
                guesses=sess.run([error],feed_dict=feed_dict)
                #print(guesses)
                i[...]=guesses[0]
                yield guesses
        errorL=min(trainFunc(),key=lambda hhh:hhh[0])[0]
        #print(errorL)
        if returnFunc:
            def _estimateFunc(*data):
                feed_dict={p:q.ravel()[...,None] for p,q in zip(oldInputs,data)}
                guesses=sess.run([gu], feed_dict=feed_dict)[0]
                return np.clip(guesses,-3,3)
            funcs.append(_estimateFunc)
    if returnFunc:
        def estimateFunc(*data):
                
            logodd=sum(f(*data) for f in funcs)
            return logodd#{"log odds": logoddprior, "prob":sigmoid(logoddprior)}
            
    emin=errorsNp.min(1)
    finalDict={"all":errorsNp, "mininum":emin, "avgmin":emin.mean(),"stdmin":emin.std()}
    if returnFunc:
        finalDict["func"]=estimateFunc
    return finalDict
def RecursivePDF(data):
    if data.shape[1]==1:
        data=data[:,0]
        isc=3.
        k=np.histogram(data,data.shape[0],(-isc,isc))
        b=(k[0]+0.00000001).astype(float)
        blur=np.exp(-np.linspace(-4,4,10)**2)
        
        b=np.convolve(b,blur,"same")
        b/=b.sum(keepdims=True)
        #plt.plot(blur)
        #plt.show()
        xp=(k[1][1:]+k[1][:-1])/2
        def f(sample):
            return invSig(np.interp(sample,xp,b))
        
    else:
        
        sp=data.shape[1]//2
        x=data[:,:sp]
        y=data[:,sp:]
        e=correlate(Data(x,normal=True),Data(y,second=True,normal=True),returnFunc=True)
        rx=RecursivePDF(x)
        ry=RecursivePDF(y)
        plt.plot(rx(np.linspace(-5,5,100)))
        plt.show()
        def f(sample):
            xs=sample[:,:sp]
            ys=sample[:,sp:]
            return rx(xs)+ry(ys)+e["func"](xs,ys)
    return f  
if __name__=="__main__":
    #dsize=DSize(.6)
    isc=2.
    dsz=300
    c=np.random.random(size=[dsz])*6.28
    x=c#np.sin(c)
    y=np.cos(c)+np.random.random(size=[dsz])*0.1
    x=normalize(x)(x)
    y=normalize(y)(y)
    v=np.meshgrid(*[np.linspace(-isc,isc,100)]*2)
    r=RecursivePDF(np.stack([x,y],axis=1))
    h=r(np.stack([v[0].ravel(),v[1].ravel()],axis=1)).reshape(100,100)
##    datax=dsize.data(x)
##    datay=dsize.data(y,second=True)
##    xn=datax.normalizer(x)
##    yn=datay.normalizer(y)
##    e=correlate(datax,datay)
##    r=e["x"]
##
##
##    
##                #print(list(i.shape for i in feed_dict.values()))
##    g=r[0].run([r[1]],feed_dict=feed_dict)
##    h=g[0].reshape(100,100)
##    betax=(np.histogram(xn,100,(-isc,isc))[0]+1).astype(float)
##    betax=np.convolve(betax,np.ones([5]),"same")
##    betax/=betax.mean(keepdims=True)
##
##    betay=(np.histogram(yn,100,(-isc,isc))[0]+1).astype(float)
##    betay=np.convolve(betay,np.ones([5]),"same")
##    betay/=betay.mean(keepdims=True)
##
##
##    h[0]=1
##    h[:,0]=1
##    h*=betax[None,:]
##    h*=betay[:,None]
    
    plt.imshow(np.clip(h,-20,20),extent=(-isc,isc,isc,-isc),interpolation="nearest")
    plt.plot(x,y,"xk")
    plt.show()
    assert 0
    #a=np.random.normal(size=[100])
    #b=np.random.normal(size=[100])
    #e=correlate(dsize.data(np.concatenate([a,b])),dsize.data(np.concatenate([a,-b]),second=True))
    #assert 0
    #e=correlate(dsize.data(np.arange(300,dtype=np.float32)),dsize.data(np.arange(300,dtype=np.float32),second=True))
    from scipy import misc
    inpu=misc.imread("TestCorrelateData7.png",flatten=True).T
    outpu=np.linspace(-1,1,150)[:,None]
    e=correlate(dsize.data(outpu),dsize.data(inpu,second=True),iterations=300)    
    #e=correlate(dsize.data(outpu),dsize.convData(inpu,[5,10,20],[10,10,10],[20,20],second=True))
    print(e)
    #print(deepCorrelate(inpu,outpu))
