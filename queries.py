import pymongo
import pandas as pd
import nltk
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import seaborn as sns
from scipy import stats
import numpy as np
import statsmodels.stats.api as sms
from statistics import mean
from scipy.spatial import distance
import xgboost as xgb
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


#connect to client 
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
print(myclient.list_database_names())
mydb = myclient["hotels"]
print(mydb.list_collection_names())
collections = mydb.list_collection_names()



#function for showing all of a key in the db 
def print_All_FromDB(target):
    for col in collections:
        mycol = mydb[col]
        docs = mycol.find()
        for doc in docs:
            print(doc[target])

#function for showing name and a target of choosing 
def print_All_FromDB_withName(target):
    for col in collections:
        mycol = mydb[col]
        docs = mycol.find()
        for doc in docs:
            print(doc['Name'] , ',' ,doc[target])

#define another function 
def print_price_ratings(target = 'Price', target_2 = 'Rating'):
    for col in collections:
        mycol = mydb[col]
        docs = mycol.find()
        for doc in docs:
            print(doc[target], ',',doc[target_2])

#call functions 
print_All_FromDB('Address')
print_All_FromDB_withName('Price')
print_price_ratings()

#very important function that takes data from mongodb and stores in a pandas frame 
def createDF(data):
    mycol = mydb[str(data)]
    docs = mycol.find()
    myData = []
    for doc in docs:
        review = doc['individual_Reviews']['Reviews']
        rating = doc['individual_Reviews']['Ratings']
        myData.append([review, rating])
    structure = {'Reviews': myData[0][0], 'Ratings': myData[0][1]}
    return(pd.DataFrame(structure))

#get data for question hw question 
def makeList_price_rating(target_1, target_2):
    global priceList 
    priceList = []
    global ratingList 
    ratingList = []
    for col in collections:
        mycol = mydb[col]
        docs = mycol.find()
        for doc in docs:
            priceList.append(doc[target_1])
            ratingList.append(doc[target_2])

        
#create plot and save data to csv 
makeList_price_rating('Price','Rating')
priceRating = pd.DataFrame({'Price': priceList, 'Ratings': ratingList})
f1 = plt.figure(1)
plt.scatter(priceRating['Ratings'], priceRating['Price'])
priceRating.to_csv('priceRating.csv', index=False)

#create frames 
ritzLondon = createDF('Ritz_london')
costaGreece = createDF('Costa_greece')
hollyWood = createDF('Hollywood')
altaC = createDF('Alta_Cienga')
grandParis = createDF('Paris_Le_Grand')
grandDelMar = createDF('Grand_Del_Mar')
hongKong = createDF('The_Peninsula_Hong_Kong')
boraBora = createDF('Bora_Bora')
monteCarlo = createDF('Monte_Carlo')
costMesa = createDF('Costa_Mesa')
nyc4S = createDF('4S_NYC')
harborInn = createDF('Harborview_Inn')
plazaInn = createDF('plaza_Inn')
bcLondon = createDF('BC_london')
starLite = createDF('star_Lite')
blvdHotel = createDF('blvd_Hotel')
bwBrook = createDF('bw_brook')
balBoa = createDF('balboa')
kingsInn = createDF('kingsInn')
sf_MexCity = createDF('sf_MC')



#add the kaggle data 
kaggleSet = pd.read_csv('tripadvisor_hotel_reviews.csv')

#combine all frames 
rawFile = pd.concat([kaggleSet, sf_MexCity, kingsInn, balBoa, bwBrook, blvdHotel , starLite, bcLondon, plazaInn, ritzLondon, costaGreece, hollyWood,altaC, grandParis,grandDelMar,hongKong,boraBora, monteCarlo,costMesa,nyc4S,harborInn], ignore_index=True)

masterWords = []
print(rawFile.head())

rowCount = len(rawFile['Ratings'])

#append everything to master words 
for elems in rawFile['Reviews']:
    masterWords.append(elems)

#create sentiment analysis 
sid = SentimentIntensityAnalyzer()

negative = []
neutral = []
positive = []
compound = []

#loop through all reviews and append their sentiment scores to new lists
for review in masterWords:
    ss = sid.polarity_scores(review)
    negative.append(ss['neg'])
    neutral.append(ss['neu'])
    positive.append(ss['pos'])
    compound.append(ss['compound'])

#get stop words
stop_words = set(stopwords.words('english')) 



#count vectorize everything and store back in pandas frame 
vec = CountVectorizer()
X = vec.fit_transform(masterWords)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df.head())



#view the mean standard deviation and the quantiles, use this to reduce columns 
#59k predictors means we have an underidentified model, more predictors than observations 
#we need to handle this somehow 
print(mean(df.std()))
print(df.std().quantile([.25,.5,.75]))
plt.hist(df.std())


#view shape and remove all columns whos std is below .025
print(df.shape)
df = df.loc[:, df.std() > 0.025]
print(df.shape)


#remove remaining stopwords 
colNames = df.columns 
print(df.shape)
goodCols = []
for i in range(0, len(colNames),1):
    if colNames[i] not in stop_words:
        goodCols.append(i)

df = df.iloc[:,goodCols]
print(df.shape)


df['Ratings'] = rawFile['Ratings']


#add the sentiments to the dataframe 
df['negScore'] = negative
df['posScore'] = positive 
df['neuScore'] = neutral 
df['compScore'] = compound


#create boxplot grouped by ratings 
f2 = plt.figure(2)
sns.boxplot(y='compScore', x='Ratings', 
                 data=df, 
                 palette="colorblind")




#main research question: How similar are adjacent ratings? Are they even different?

#find means and variance
groupMean = df.groupby('Ratings')['compScore'].mean()          
print(groupMean)

groupSD = df.groupby('Ratings')['compScore'].std()          
print(groupSD)




#run significance tests to understand differences between groups
#check if the composite sentiment score is truly different for adjacent rating groups
result_5_4 = stats.ttest_ind( df.loc[df['Ratings'] == 5, 'compScore'], df.loc[df['Ratings'] == 4, 'compScore'], equal_var = False)
print(result_5_4)

#is 4 different to 3
result_4_3 = stats.ttest_ind( df.loc[df['Ratings'] == 4, 'compScore'], df.loc[df['Ratings'] == 3, 'compScore'], equal_var = False)
print(result_4_3)

#is 3 different to 2
result_3_2 = stats.ttest_ind( df.loc[df['Ratings'] == 3, 'compScore'], df.loc[df['Ratings'] == 2, 'compScore'], equal_var = False)
print(result_3_2)

#is 2 different to 1
result_2_1 = stats.ttest_ind( df.loc[df['Ratings'] == 2, 'compScore'], df.loc[df['Ratings'] == 1, 'compScore'], equal_var = False)
print(result_2_1)


#we can do some confidence intervals around means 
conf = sms.DescrStatsW(df.loc[df['Ratings'] == 5, 'compScore']).tconfint_mean()
conf_2 = sms.DescrStatsW(df.loc[df['Ratings'] == 4, 'compScore']).tconfint_mean()
print(conf)
print(conf_2)


#extract vectors 
vector5 = df.loc[df['Ratings'] == 5, 'compScore']
vector4 = df.loc[df['Ratings'] == 4, 'compScore']
vector3 = df.loc[df['Ratings'] == 3, 'compScore']
vector2 = df.loc[df['Ratings'] == 2, 'compScore']
vector1 = df.loc[df['Ratings'] == 1, 'compScore']

#calculate the cosine similarity between two adjacent ratings 
#-1 strong opposite vector  
#0 orthogonal to each other 
#1 strong similarity 
sim_5_4 = cosine_similarity([vector5[:2000]], [vector4[:2000]])
print('similarity between ratings 5 and 4', sim_5_4)

sim_4_3 = cosine_similarity([vector4[:2000]], [vector3[:2000]])
print('similarity between ratings 4 and 3', sim_4_3)

sim_3_2 = cosine_similarity([vector3[:2000]], [vector2[:2000]])
print('similarity between ratings 3 and 2', sim_3_2)

sim_2_1 = cosine_similarity([vector2[:2000]], [vector1[:2000]])
print('similarity between ratings 2 and 1', sim_2_1)


#visualize ratings given compScore, negScore, and posScore in three dimensions 
X = df[['compScore', 'negScore', 'posScore']]
fig = px.scatter_3d(
    X, x='compScore', y='negScore', z='posScore', color= df['Ratings'].astype('category'),
    title=f'Ratings in 3D',
    labels={'0': 'compScore', '1': 'negScore', '2': 'posScore'}
)
#fig.show()


#based on the above, we can collapse 4 and 5 but no more than that 
df['Ratings'] = df['Ratings'].replace([4],[5])

f3 = plt.figure(3)
sns.boxplot(y='compScore', x='Ratings', 
                 data=df, 
                 palette="colorblind")

#research question: What does an outlier look like? What is an outlier in a text based setting

#whats going on with the 1 and 1 and the -1 and 5?
pd.set_option('display.max_colwidth', -1)


#bad sentiment but good review 
max_1 = (df.loc[df['Ratings'] == 5]['compScore'].idxmin())
print(rawFile.iloc[max_1,:])


#good sentiment but bad review 
#this poor review happen to come from the kaggle set which has no version with the stop words. 
max_1 = (df.loc[df['Ratings'] == 1]['compScore'].idxmax())
print(rawFile.iloc[max_1,:])


#here I want to get all the reviews under the bottom line of 3 and 4
#we need to quants to calculate cutoff for outliers
quants = df.groupby('Ratings')['compScore'].quantile([.25,.75])
print(quants)  

#calculation is the definition of an outlier as shown on the boxplot
#how many reviews are potential problems (considered outliers)
threshold_3 = 0.694800 - ((0.973100 - 0.694800)*1.5)
ids_3 = (df.index[(df['Ratings'] == 3) & (df['compScore'] < threshold_3)]).tolist()
print(len(ids_3))


threshold_5 = 0.942900 - ((0.987800 - 0.942900)*1.5)
ids_5 = (df.index[(df['Ratings'] == 5) & (df['compScore'] < threshold_5)]).tolist()
print(len(ids_5))


#it may not be truly reflective to delete these observations but I do anyway
deleteList = ids_3 + ids_5
print(len(deleteList))

df = df.drop(deleteList)

#visualize with outliers removed
f4 = plt.figure(4)
sns.boxplot(y='compScore', x='Ratings', 
                 data=df, 
                 palette="colorblind")


f5 = plt.figure(5)
sns.boxenplot(y='compScore', x='Ratings', 
                 data=df, 
                 palette="colorblind")




#sample data until the group sizes match 
#we need to rebalance for better classification results
minGroup = df['Ratings'].value_counts().min()

def sampling_k_elements(group, k = minGroup):
    if len(group) < k:
        return group
    return group.sample(k)

print(df['Ratings'].value_counts())
balanced = df.groupby('Ratings').apply(sampling_k_elements).reset_index(drop=True)
print(balanced['Ratings'].value_counts())




#visualize ratings again after data cleaning. compScore, negScore, and posScore in three dimensions 
X = balanced[['compScore', 'negScore', 'posScore']]
fig = px.scatter_3d(
    X, x='compScore', y='negScore', z='posScore', color= balanced['Ratings'].astype('category'),
    title=f'Ratings in 3D',
    labels={'0': 'compScore', '1': 'negScore', '2': 'posScore'}
)
#fig.show()



#extract ratings into its own vector outside the dataframe 
target = balanced['Ratings']
del balanced['Ratings']


#create a dataframe that ignores the lexicon scores
onlyML = balanced.drop(['negScore', 'posScore', 'neuScore', 'compScore'], axis=1)

#split into train and test 
X_train, X_test, y_train, y_test = train_test_split(balanced, target, test_size=0.30, random_state=1)
X_trainML, X_testML, y_trainML, y_testML = train_test_split(onlyML, target, test_size=0.30, random_state=1)


#create the regressor objects
#random forest with full data 
regressor = RandomForestClassifier(n_estimators=100)

#random forest without any lexicon 
regressor_ = RandomForestClassifier(n_estimators=100)


#pass in training data 
regressor.fit(X_train, y_train)

#save predictions in new list 
y_pred = regressor.predict(X_test)


#evaluate the results 
print('results of ML with sentiment scores')
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


#pass in training data 
regressor_.fit(X_trainML, y_trainML)

#save predictions in new list 
y_predML = regressor_.predict(X_testML)

#evaluate the results 
print('results of ML with no sentiment scores')
print(confusion_matrix(y_testML,y_predML))
print(accuracy_score(y_testML,y_predML))



#gradient boosting with full data, full data had better results 
regressorXGB = xgb.XGBClassifier(
    n_estimators=100,
    reg_lambda=.75,
    gamma=0,
    max_depth=5
)



#fit model, it takes a long time so pre and post build prints help see where we are 
print('preBuild')
regressorXGB.fit(X_train, y_train)
y_predXGB = regressorXGB.predict(X_test)
print('post')



#evaluate the results 
print('results of XGBoost')
print(confusion_matrix(y_test,y_predXGB))
print(accuracy_score(y_test,y_predXGB))


#save importance object 
importance = regressor.feature_importances_

cols = list(balanced.columns)
#create a new dataframe with column names and their importance level 
importanceFrame = pd.DataFrame({'columns': cols, 'importance': importance})

#sort the dataframe to answer the data question 
question1_hw2 = importanceFrame.sort_values(by=['importance'])
print(question1_hw2)
question1_hw2.to_csv('question1_hw2.csv', index=False)



#taken from online resource 
def wordCloud_generator(data, title=None):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='black',
                          min_font_size = 10
                         ).generate(" ".join(data.values))
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.title(title,fontsize=30)
    plt.show() 


#comparison 1 
#wordCloud_generator(ritzLondon['Reviews'], title="Most used words in reviews")
#wordCloud_generator(boraBora['Reviews'], title="Most used words in reviews")


#cross validate the best model 
scores = cross_val_score(regressorXGB, balanced, target, cv=3)

print(scores)
print(mean(scores))

plt.show()
