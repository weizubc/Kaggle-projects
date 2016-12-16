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

# Data cleanup
# TRAIN DATA
bids= pd.read_csv('bids.csv')

#get time lag columns to compute timediff within bidder_id&auction group
bids['timelag']=bids.groupby(["bidder_id","auction"])['time'].transform(lambda x: x.shift(1))
bids['timediff']=bids.time-bids.timelag


train = pd.read_csv('train.csv')

#pd.value_counts(train['outcome'])
#robot_id=train.loc[train['outcome']==1,:]
#robot_data=pd.merge(bids,robot_id,on='bidder_id',how='inner')
#r=bids.loc[bids['bidder_id']=='af9c96944265cf541b3fe734a057821a825l7',:]
#h=bids.loc[bids['bidder_id']=='91a3c57b13234af24875c56fb7e2b2f4rb56a',:]

combine=pd.merge(bids,train,on='bidder_id',how='right')

#group by bidder_id
group=combine.groupby(["bidder_id"],as_index=False)
agg=group.agg({"bid_id":pd.Series.count,"outcome":np.max})
agg1=agg.rename(columns={'bid_id':'nofbids'})


#group by bidder_id and auction
#on using as_index=False to get SQL like output

group=combine.groupby(["bidder_id","auction"],as_index=False)
agg=group.agg({"device":pd.Series.nunique, "url":pd.Series.nunique,"country":pd.Series.nunique,"bid_id":lambda x: x.count(),"timediff":np.mean,"merchandise": lambda x: x.iloc[0]})


group=agg.groupby(["bidder_id"],as_index=False)
agg=group.agg({"device":np.mean, "url":np.mean,"country":np.mean,"bid_id":np.mean,"timediff":np.mean,"merchandise": pd.Series.count, "auction":pd.Series.count})
agg2=agg.rename(columns={'device':'nofdev','url':'nofurl','country':'nofctry','bid_id':'avgnofbids','timediff':'avgtimediff','merchandise':'nofmer','auction':'nofauc'})


#calculate the winner for each auction
#group by auction
bids['bidder_id2']=bids['bidder_id']
group=bids.groupby(["auction"],as_index=False)
agg=group.agg({"bidder_id":lambda x: x.iloc[-1],"bidder_id2":pd.Series.nunique})
agg3=agg.rename(columns={'bidder_id':'winner_id','bidder_id2':'nofbidders'})


#group by winner_id
group=agg3.groupby(["winner_id"],as_index=False)
agg=group.agg({"nofbidders":pd.Series.count})
agg4=agg.rename(columns={'winner_id':'bidder_id','nofbidders':'nofwins'})


#group by winner_id and set nofbidders>1
agg3=agg3[agg3['nofbidders']>1]
group=agg3.groupby(["winner_id"],as_index=False)
agg=group.agg({"nofbidders":pd.Series.count})
agg5=agg.rename(columns={'winner_id':'bidder_id','nofbidders':'nofwins2'})


#combining all the summary columns
result=pd.merge(agg1,agg2,on='bidder_id',how='left')
result=pd.merge(result,agg4,on='bidder_id',how='left')
train_result=pd.merge(result,agg5,on='bidder_id',how='left')



#filling in NA values
train_result['nofdev']= train_result['nofdev'].fillna(0)
train_result['nofurl']= train_result['nofurl'].fillna(0)
train_result['nofctry']= train_result['nofctry'].fillna(0)
train_result['nofauc']= train_result['nofauc'].fillna(0)
train_result['nofwins']= train_result['nofwins'].fillna(0)
train_result['nofwins2']= train_result['nofwins2'].fillna(0)
train_result['nofmer']= train_result['nofmer'].fillna(0)
train_result['avgnofbids']= train_result['avgnofbids'].fillna(0)
train_result['avgtimediff']= train_result['avgtimediff'].fillna(bids['timediff'].max())

train_result['winpct']=train_result.nofwins/train_result.nofauc
train_result['winpct']= train_result['winpct'].fillna(0)




# TEST DATA
test = pd.read_csv('test.csv')

combine=pd.merge(bids,test,on='bidder_id',how='right')

#group by bidder_id
group=combine.groupby(["bidder_id"],as_index=False)
agg=group.agg({"bid_id":pd.Series.count})
agg1=agg.rename(columns={'bid_id':'nofbids'})


#group by bidder_id and auction
#on using as_index=False to get SQL like output

group=combine.groupby(["bidder_id","auction"],as_index=False)
agg=group.agg({"device":pd.Series.nunique, "url":pd.Series.nunique,"country":pd.Series.nunique,"bid_id":lambda x: x.count(),"timediff":np.mean,"merchandise": lambda x: x.iloc[0]})


group=agg.groupby(["bidder_id"],as_index=False)
agg=group.agg({"device":np.mean, "url":np.mean,"country":np.mean,"bid_id":np.mean,"timediff":np.mean,"merchandise": pd.Series.count, "auction":pd.Series.count})
agg2=agg.rename(columns={'device':'nofdev','url':'nofurl','country':'nofctry','bid_id':'avgnofbids','timediff':'avgtimediff','merchandise':'nofmer','auction':'nofauc'})


#combining all the summary columns
result=pd.merge(agg1,agg2,on='bidder_id',how='left')
result=pd.merge(result,agg4,on='bidder_id',how='left')
test_result=pd.merge(result,agg5,on='bidder_id',how='left')


#filling in NA values
test_result['nofdev']=test_result['nofdev'].fillna(0)
test_result['nofurl']=test_result['nofurl'].fillna(0)
test_result['nofctry']=test_result['nofctry'].fillna(0)
test_result['nofauc']=test_result['nofauc'].fillna(0)
test_result['nofwins']=test_result['nofwins'].fillna(0)
test_result['nofwins2']=test_result['nofwins2'].fillna(0)
test_result['nofmer']=test_result['nofmer'].fillna(0)
test_result['avgnofbids']=test_result['avgnofbids'].fillna(0)
test_result['avgtimediff']=test_result['avgtimediff'].fillna(bids['timediff'].max())
test_result['winpct']=test_result.nofwins/test_result.nofauc
test_result['winpct']= test_result['winpct'].fillna(0)


predictors = ["nofdev", "nofurl", "avgnofbids", "avgtimediff", "nofmer", "nofauc", "nofwins"]
X=train_result[predictors]
y=train_result['outcome']




from sklearn.feature_selection import SelectKBest, f_classif
# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(X,y)

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
import matplotlib.pyplot as plt
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

train_result.groupby(['outcome']).avgnofbids.mean()


alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg,X,y,cv=3)
print(scores.mean())
print(scores)


alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg,X,y,cv=3)
print(scores.mean())
print(scores)






#ensemble
algorithms = [
              [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2), predictors],
              [LogisticRegression(random_state=1), predictors]
              ]

# Initialize the cross validation folds
kf = KFold(train_result.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = train_result["outcome"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(train_result[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(train_result[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0]>0.5) | (full_test_predictions[1]>0.5)
    test_predictions=test_predictions.astype(float)
    predictions.append(test_predictions)


# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions == train_result["outcome"]).astype(float) / len(predictions)
print(accuracy)




#submission
model = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
model.fit(X,y)
predictions = model.predict(test_result[predictors]).astype(int)


submission = pd.DataFrame({
                           "bidder_id":test_result['bidder_id'],
                           "predictions": predictions
                              })


submission.to_csv("kaggle.csv", index=False)
