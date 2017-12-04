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
    """When called,  returns cycled data followed by plain data.
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
def makeConvModel(dataShape,convFilters,  convKernal,  layers,convDim=1):
    """ returns a keras sequential model. This must be done repeatedly
    because their is no way to reset the model to random starting values
    """
    assert convDim in (1,2,3)
    ConvLayer=[None,Conv1D,Conv2D,Conv3D][convDim]
    
    assert len(dataShape) in (3,4)
    while True:
        model=Sequential()
        model.add(ConvLayer(convFilters[0],convKernal[0],input_shape=dataShape[1:],padding="same",activation="relu"))
        for i in zip(convFilters[1:],convKernal[1:]):
            model.add(ConvLayer(*i,padding="same",activation="relu"))
        model.add(Lambda(lambda x: tf.reduce_mean(x,axis=list(range(1,convDim+1)))))
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
##        """data: a numpy ndarray of data,  layers =[5,5] describes depth
##        and bredth of neural net ,second:controls what correlates to
##        what.
##        """
##        return Data(self,*args,**kwargs)
##    def convData(self,*args,**kwargs):
##        """data: a numpy ndarray of data,  layers =[5,5] describes depth
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
    
    def __init__(self,data,split=None,convFilters=[3,3],  convKernal=[4,5],  layers=[5],convDim=2,second=False,normal=False):
        assert len(convFilters)==len(convKernal)
        if split==None:
            split=data.shape[0]-1
        if split<1:
            split=int(split*data.shape[0])
        self.lenTrain=split
        self.lenTest=data.shape[0]-split
        while data.ndim<2+convDim:
            data=data[...,None]

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
        
        self.model=makeConvModel(data.shape,convFilters,  convKernal,  layers,convDim)

def correlate(*data,repeats=1,iterations=1,returnFunc=False,train=.6):
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
    for thisErrors in errorsNp:#np.nditer(errorsNp,  op_flags=['readwrite']):#www in range(repeats):
        print(thisErrors.shape)
        oldModels=[next(i.model) for i in data]
        oldOutputs=[i.output for i in oldModels]
        oldInputs=[i.input for i in oldModels]
        colapsed=tf.concat(oldOutputs,1)
        model=Sequential()
        model.add(Dense(20,input_shape=[int(colapsed.shape[1])],activation="relu"))
        model.add(Dense(20,activation="relu"))
        model.add(Dense(1))
        outputTf=tf.placeholder(tf.float32,  shape=(None,1))
        gu=model(colapsed)
        error=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=outputTf,logits=gu))/np.log(2)
        train=tf.train.AdamOptimizer(.04).minimize(error)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        x=np.zeros([iterations])#iterations
        
        def trainFunc():
            for i in np.nditer(thisErrors,  op_flags=['readwrite']):
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
                
                feed_dict={p:q for p,q in zip(oldInputs,data)}
                guesses=sess.run([gu],  feed_dict=feed_dict)[0]
                return np.clip(guesses,-3,3)
            funcs.append(_estimateFunc)
    if returnFunc:
        def estimateFunc(*data):
                
            logodd=sum(f(*data) for f in funcs)
            return logodd#{"log odds": logoddprior,  "prob":sigmoid(logoddprior)}
            
    emin=errorsNp.min(1)
    finalDict={"all":errorsNp,  "mininum":emin,  "avgmin":emin.mean(),"stdmin":emin.std()}
    if returnFunc:
        finalDict["func"]=estimateFunc
    return finalDict
def RecursivePDF(data):
    dataSize=np.product(data.shape[1:])
    if dataSize ==1:
        data=data.ravel()
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
    elif dataSize <=16:
        data=data.reshape(-1,dataSize)
        sp=dataSize[1]//2
        x=data[:,:sp]
        y=data[:,sp:]
        e=correlate(Data(x,normal=True),Data(y,second=True,normal=True),returnFunc=True)
        rx=RecursivePDF(x)
        ry=RecursivePDF(y)
        def f(sample):
            
            xs=sample[:,:sp]
            ys=sample[:,sp:]
            return rx(xs)+ry(ys)+e["func"](xs,ys)
    else:
        b=data.shape[1]>data.shape[2]
        if b:
            sp=data.shape[1]//2
            x=data[:,:sp]
            y=data[:,sp:]
            
        else:
            sp=data.shape[2]//2
            x=data[:,:,:sp]
            y=data[:,:,sp:]
            
        e=correlate(ConvData(x,normal=True),ConvData(y,second=True,normal=True),returnFunc=True)
        rx=RecursivePDF(x)
        ry=RecursivePDF(y)
        #plt.plot(rx(np.linspace(-5,5,100)))
        #plt.show()
        def f(sample):
            if b:
                xs=sample[:,:sp]
                ys=sample[:,sp:]
            else:
                xs=sample[:,:,:sp]
                ys=sample[:,:,sp:]
            return rx(xs)+ry(ys)+e["func"](xs,ys)
    return f  
if __name__=="__main__":
    #dsize=DSize(.6)
    hilbert=np.array([[21,22,25,26,37,38,41,42],
                      [20,23,24,27,36,39,40,43],
                      [19,18,29,28,35,34,45,44],
                      [16,17,30,31,32,33,46,47],
                      [15,12,11,10,53,52,51,48],
                      [14,13, 8, 9,54,55,50,49],
                      [ 1, 2, 7, 6,57,56,61,62],
                      [ 0, 3, 4, 5,58,59,60,63]],int).ravel()
    
    k=np.zeros((64,),int)
    k[hilbert]=np.arange(64)
    from sklearn import datasets;
    digits = datasets.load_digits()
    train=digits.images[:898,...,None]
    test=digits.images[898:1796,...,None]
    imgs=[i for i in (train,test)]
    
    rimgs=[i.swapaxes(1,2) for i in (train,test)]
   
    vimgs=[i[:,::-1] for i in (train,test)]
    norm=normalize(imgs[0])
    
    
    #print(img.shape,rimg.shape)
##    isc=2.
##    dsz=300
##    c=np.random.random(size=[dsz])*6.28
##    x=c#np.sin(c)
##    y=np.cos(c)+np.random.random(size=[dsz])*0.1
##    x=normalize(x)(x)
##    y=normalize(y)(y)
##    v=np.meshgrid(*[np.linspace(-isc,isc,100)]*2)
##    r=RecursivePDF(np.stack([x,y],axis=1))
##    h=r(np.stack([v[0].ravel(),v[1].ravel()],axis=1)).reshape(100,100)
    r=RecursivePDF(imgs[0])
    x=[r(norm(i))[:,0] for i in imgs]
    y=[r(norm(i))[:,0] for i in rimgs]
    v=[r(norm(i))[:,0] for i in vimgs]
    q=digits.target[898:1796]
    z=[v[1][q==i] for i in range(10)]
    [i.sort() for i in z]
    [i.sort() for i in x+y+v]

    [plt.plot(np.linspace(0,1,i.size),i,"#"+j) for i,j in zip(z,"800000,404000,008000,004040,000080,400040,000000,404040,808080,c0c0c0".split(','))]
    plt.show()
    #[i.mean() for i in z]
    #[-298.23948981760128, -270.53660615117275, -323.51299285366628, -328.30029981830688, -250.95364090116499, -315.29797660800233, -279.69289920493873, -280.24501667738627, -309.65449304957366, -308.33006161553914]
    plt.plot(x[0],"#000000")
    plt.plot(x[1],"#404040")
    plt.plot(y[0],"#000080")
    plt.plot(y[1],"#4040C0")
    plt.plot(v[0],"#800000")
    plt.plot(v[1],"#C04040")
    plt.show()
    

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
    
    #plt.imshow(np.clip(h,-20,20),extent=(-isc,isc,isc,-isc),interpolation="nearest")
    #plt.plot(x,y,"xk")
    #plt.show()
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
