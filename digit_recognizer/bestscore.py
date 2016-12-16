import numpy as np
import pandas as pd
from nolearn.dbn import DBN
from scipy.ndimage import convolve
import scipy.ndimage as nd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
###############################################################################
# Setting up

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
                     [[0, 1, 0],
                      [0, 0, 0],
                      [0, 0, 0]],[[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 0]],[[0, 0, 0],
                                              [0, 0, 1],
                                              [0, 0, 0]],[[0, 0, 0],
                                                          [0, 0, 0],
                                                          [0, 1, 0]]]
    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant', weights=w).ravel()
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

def rotate_dataset(X):
    XX = np.zeros(X.shape)
    for index in range(X.shape[0]):
        angle = np.random.randint(-7,7)
        XX[index,:] = nd.rotate(np.reshape(X[index,:],((28,28))),angle,reshape=False).ravel()
    return XX

# Load Data
mnist = pd.read_csv("train.csv")
y_train = mnist['label'].values
X_train = mnist.loc[:,'pixel0':].values
X_test = pd.read_csv("test.csv").values
X_train = np.asarray(X_train / 255.0, 'float32')
X_test = np.asarray(X_test / 255.0, 'float32')
X_train, y_train = nudge_dataset(X_train, y_train)
X_train= rotate_dataset(X_train)

clf = DBN(
                                                             [X_train.shape[1], 1000, 10],
                                                             learn_rates=0.3,
                                                             learn_rate_decays=0.95,
                                                             learn_rates_pretrain=0.005,
                                                             epochs=120,
                                                             verbose=1,
                                                             )

clf.fit(X_train, y_train)
save= clf.predict(X_test)





# Data cleanup
# TRAIN DATA
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine=pd.concat([train,test],keys=['train','test'])

X=combine.iloc[:,1:]

#scale X with no elimination
avg=X.mean()
std=X.std()
#pd.value_counts(std)

def redefine(a):
    if a==0:
        return 1
    return a

std=std.apply(redefine)
X_scale=(X-avg)/std



#eliminate zero-variance columns
std=X.std()
zero=(std==0)
X=X.drop(X.columns[zero],axis=1)


#scale X with elimination
avg=X.mean()
std=X.std()
X_scale0=(X-avg)/std


#do pca dimension reduction to 200
pca = PCA(n_components=200)
X_transform=pca.fit_transform(X_scale)



#separate train data from combine
y=combine['label'].ix['train']
train_transform=X_transform[0:len(y)]




#find the element with highest counts
def mode(a):
    count=0
    element=a[0]
    for i in a:
        if a.count(i)>count:
            count=a.count(i)
            element=i
    return element



#reorgnize list
#a=list([array2,array2,array3])
def reorg(a):
    element=[]
    for i in range(len(a[0])):
        b=[]
        for j in range(len(a)):
            b.append(a[j][i])
        element.append(mode(b))
    return element




#submission
test_transform=X_transform[len(y):]
id=np.array(list(range(len(test_transform))))+1



algorithms = [
              [svm.SVC(random_state=1, kernel='poly',degree=3),train_transform,y,test_transform],
              [svm.SVC(random_state=1, kernel='poly',degree=2),train_transform,y,test_transform],
              [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2),X_scale0.ix['train'],y,X_scale0.ix['test']],
              [MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(1000,500), random_state=1),X_scale0.ix['train'],y,X_scale0.ix['test']]
                 ]


full_predictions = []
full_predictions.append(save)
for alg,data,datatarget,test in algorithms:
    alg.fit(data,datatarget)
    test_predictions = alg.predict(test).astype(int)
    full_predictions.append(test_predictions)


predictions = reorg(full_predictions)



submission = pd.DataFrame({
                          "ImageId":id,
                          "Label": predictions
                          })


submission.to_csv("kaggle.csv", index=False)

#0.98443 combining with other algorithms
#Random forest around 0.962 accuracy
#SVM around 0.973 accuracy with degree 3 work best(need to do pca to speed up)
#Neural network without expanding data around 0.966 accuracy
