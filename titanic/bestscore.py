import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import svm
import numpy as np
import re
from sklearn.linear_model import LinearRegression
import operator
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


titanic = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")
combine=pandas.concat([titanic,titanic_test],keys=['train','test'])

combine.loc[combine["Sex"] == "male", "Sex"] = 0
combine.loc[combine["Sex"] == "female", "Sex"] = 1

#print(combine["Embarked"].unique())
combine['Embarked']=combine['Embarked'].fillna('S')
combine.loc[combine['Embarked'] == 'S', 'Embarked'] = 0
combine.loc[combine['Embarked'] == 'C', 'Embarked'] = 1
combine.loc[combine['Embarked'] == 'Q', 'Embarked'] = 2
combine['Fare'] = combine['Fare'].fillna(combine['Fare'].median())


# Generating a familysize column
combine["FamilySize"] = combine["SibSp"] + combine["Parch"]

# The .apply method generates a new series
combine["NameLength"] = combine["Name"].apply(lambda x: len(x))


# Generating a title column

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = combine["Name"].apply(get_title)
#print(pandas.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# Verify that we converted everything.
#print(pandas.value_counts(titles))

# Add in the title column.
combine["Title"] = titles



#genearting a married dummy
combine["Married"]=0
combine.loc[(combine['Title']==1) | (combine['Title']==3), 'Married'] =1



#predict age for missing values
predictors = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]
alg = LinearRegression()

y=combine['Age'].dropna()
missing=pandas.isnull(combine['Age'])
x=combine[predictors][-missing]
y=combine["Age"][-missing]
alg.fit(x,y)
predictions=alg.predict(combine[predictors][missing])

#combine.loc[missing, 'Age'] = predictions
#combine['Age'] = combine['Age'].fillna(combine['Age'].median())

#use the median age of the title group as best guess
a=combine.groupby(["Title"]).Age.median()

# A function to get the number of women in a family given a row
def get_age(row):
    if pandas.isnull(row['Age']):
        row['Age']=a[row['Title']]
    return row

# Get the number of women with the apply method
combine = combine.apply(get_age, axis=1)



#Generating a family id column

# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Get the family ids with the apply method
id = combine.apply(get_family_id, axis=1)
combine["id"] = id

#save to identify the family
family_ids=id*1

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[combine["FamilySize"] < 3] = -1

# Print the count of each unique id.
#print(pandas.value_counts(family_ids))

combine["FamilyId"] = family_ids




#Generating the number of women in a family
combine["Sex"]=combine["Sex"].astype(float)
a=combine.groupby(["id"]).Sex.sum()

# A function to get the number of women in a family given a row
def get_sum(row):
    id_ask=row["id"]
    return a.loc[id_ask]

# Get the number of women with the apply method
women = combine.apply(get_sum, axis=1)
combine["FamilyWomen"]=women




#Generating the number of children in a family
combine["child"]=0
combine.loc[combine['Age'] < 18, 'child'] = 1
a=combine.groupby(["id"]).child.sum()

# A function to get the number of children in a family given a row
def get_sum(row):
    id_ask=row["id"]
    return a.loc[id_ask]

# Get the number of children with the apply method
children= combine.apply(get_sum, axis=1)
combine["FamilyChildren"]=children





#Genearting cabin feature variable

#print(combine["Cabin"].unique())
#print(pandas.value_counts(combine["Cabin"]))
#combine.loc[pandas.isnull(combine['Cabin']), 'Cabin'] = 0

combine['Cabin']=combine['Cabin'].fillna('UNK')

#print(pandas.value_counts(combine["Cabin"]))

def get_letter(name):
    # Use a regular expression to search
    letter_search = re.search('^[A-Z]+', name)
    # If the letter exists, extract and return it.
    if letter_search:
        return letter_search.group(0)
    return ""

# Get all the letters and print how often each one occurs.
combine["Cabin"]=combine["Cabin"].astype(object)
letter = combine["Cabin"].apply(get_letter)
#print(pandas.value_counts(letter))

letter_mapping = {"UNK": 1, "C": 2, "B": 3, "D": 4, "E": 5, "A": 6, "F": 7, "G": 8, "T": 9}
for k,v in letter_mapping.items():
    letter[letter == k] = v

combine["CabinLetter"] = letter





#Generating survivor feature variable
#if other family members survived
#filling the survived value=0 for test data temporarily
combine['Survived'] = combine['Survived'].fillna(0)
a=combine.groupby(["id"]).Survived.sum()

def get_sum(row):
    id_ask=row["id"]
    return a.loc[id_ask]

# Get the number of survivors with the apply method
survivor = combine.apply(get_sum, axis=1)
combine['FamilySurvive']=survivor-combine['Survived']




#if other cabin members survived
a=combine.groupby(["Cabin"]).Survived.sum()

def get_sum(row):
    return a.loc[row["Cabin"]]

# Get the number of survivors with the apply method
survivor = combine.apply(get_sum, axis=1)
combine['CabinSurvive']=survivor-combine['Survived']
combine.loc[combine['Cabin']=='UNK', 'CabinSurvive'] =0



#Genearating help dummy
combine['helpdummy']=0
combine.loc[((combine['FamilySurvive']>0) | (combine['CabinSurvive']>0)) & (combine['Sex']==1), 'helpdummy'] = 1
combine.loc[((combine['FamilySurvive']>0) | (combine['CabinSurvive']>0)) & (combine['Age']<18), 'helpdummy'] = 1




#train data
titanic=combine.ix['train']

tpredictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId","Married","helpdummy"]
lpredictors=["Pclass", "Sex", "Fare", "Parch","SibSp", "Title", "Age", "Embarked","FamilyId","FamilyWomen","CabinLetter","NameLength","Married","helpdummy"]
spredictors=["Pclass", "Sex", "Age", "FamilySize", "Fare", "Embarked", "Title","FamilyId","FamilyWomen"]
# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=6, min_samples_leaf=3)
scores = cross_validation.cross_val_score(alg,titanic[tpredictors],titanic["Survived"],cv=3)
print(scores.mean())
print(scores)





#ensemble

algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=6, min_samples_leaf=3), tpredictors],
    [LogisticRegression(random_state=1), lpredictors]
]


# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.  
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    #predict suvivial=1 if both models have predict_prob>0.5
    test_predictions = (full_test_predictions[0]>0.5) & (full_test_predictions[1]>0.5)
    test_predictions=test_predictions.astype(float)
    predictions.append(test_predictions)


# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions == titanic["Survived"]).astype(float) / len(predictions)
print(accuracy)





#submission
titanic_test=combine.ix['test']

algorithms = [
              [RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=6, min_samples_leaf=3), tpredictors],
              [LogisticRegression(random_state=1), lpredictors]
               ]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

predictions = (full_predictions[0]>0.5) & (full_predictions[1]>0.5)
predictions=predictions.astype(int)
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })


print(pandas.value_counts(predictions))

submission.to_csv("kaggle.csv", index=False)
