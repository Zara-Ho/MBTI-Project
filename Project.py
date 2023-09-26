#Project

# Data Analysis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
# Data Visualizatioin
import matplotlib.pyplot as plt
import seaborn as sns #similar to matplotlib but can create more advanced statistical plots
# Train Test
from sklearn import model_selection
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer 

# Importing Dataset
df = pd.read_csv('C:/Users/user/Downloads/mbti_1.csv/mbti_1.csv')
print(df.head(5)) 


# Exploring Dataset (Data Analysis)
print(df.isnull().any()) #indicates if there is any null values in the dataset
print("Since both type and post are False, there is no null value in the dataset")
row,col = df.shape
print(f'There are {row} rows and {col} columns') #list out the number of rows and columns
print(df.describe(include=['object'])) 
print('There are 16 unique type, with INFP being the type with the most frequncy of 1832 times')
# shows the different types of personalities in the sample
print("Unique values from the column type =",df['type'].unique())

#Find the total number of post for each type
post_number_each_type = df.groupby(['type']).count()*50
print(post_number_each_type) #multiplied by 50 because each has its last 50 comments 
#Plot the post_number_each_type data
total = post_number_each_type.sort_values("posts", ascending=False)
plt.figure(figsize = (10,4))
sns.barplot(x=total.index, y=total.posts)
plt.xlabel('Type')
plt.ylabel('Total amount of posts')
plt.title('Total Number of Post for each Personality Type')
plt.show()

#Find the length of all 50 comments per user
df['Post_Length'] = df['posts'].apply(len)
sns.displot(df['Post_Length'], kde=True)
plt.title("Length of the 50 posts for each user") # Mostly between 70000-9000 words
plt.show()


#Clean the most common words in the last 50 comments
df['no_link_posts'] = df['posts'].replace('\|\|\|',' ', regex=True)
df['no_link_posts'] = df["no_link_posts"].str.replace(r'https?://[^\s<>"]+|www\.[^\s<>"]+', "", regex=True) #Remove links
df['no_link_posts'] = df['no_link_posts'].str.replace(r'[^a-zA-Z]',' ', regex=True) #Remove all symbols
df['no_link_posts'] = df['no_link_posts'].str.replace(r'\b\w{1,3}\b', '', regex=True) #Remove words that are too short
df['no_link_posts'] = df['no_link_posts'].str.replace(r'\b\w{30,10000}\b', '', regex=True) #Remove words that are too long
df['no_link_posts'] = df['no_link_posts'].str.lower() #Replace all letters in lowercase
# Remove MBRI words
mbti_16 = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
mbti_16 = [p.lower() for p in mbti_16]
p = re.compile("(" + "|".join(mbti_16) + ")")
df['no_link_posts'] = df['no_link_posts'].str.replace(p, '', regex=True)
stopword = stopwords.words('english') #Finding for stopwords (useless words for word processing)
df['filtered_posts'] = df['no_link_posts'].apply(lambda y: ' '.join([word for word in y.split() if word not in stopword]))

#Find the most common words used 
post_split = list(df["filtered_posts"].apply(lambda x: x.split())) #Split based on a spacebar
post_split = [x for y in post_split for x in y]
most_occur = Counter(post_split).most_common(40)
print(most_occur)
#Plot the most common words used for all MBTIs
# Collocation = False helps to get rid of word duplicates and words that frequently group tgt
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(" ".join(post_split)) 
plt.figure(figsize = (15,10))
plt.imshow(word_cloud)
plt.title('The most common words used for ALL MBTIs')
plt.axis("off")
plt.show()

#Plot the most common words used for each MBTI
j = 0
for i in df['type'].unique():
    mbti = df[df['type'] == i]   
    word_cloud1 = WordCloud(collocations = False, background_color = 'white').generate(mbti['filtered_posts'].to_string())
    plt.subplot(4,4,j+1)   
    plt.imshow(word_cloud1)
    plt.title("The most common words used for EACH MBTI")
    plt.title(i)
    plt.axis('off')
    j+=1


# Machine Learning 
df['encode'] = df['type'].astype('category').cat.codes #x = assign each MBTI with a code
df['word_number'] = df['filtered_posts'].apply(lambda z: len(z.split())) #y = number of words in each row after cleaning
vectorizer = CountVectorizer() 
X = vectorizer.fit_transform(df["filtered_posts"])
y = df.encode
print(X.shape)
print(y.shape) #8675 samples and 92820 features
np.random.seed(123) #set random seed to get the same random numbers
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.7)
print(X_train.shape)
Accuracy = {}
#Decision Trees Classifer
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
print(metrics.classification_report(y_test, y_test_pred))
print("Accuracy Score :", accuracy_score(y_test, y_test_pred))
Accuracy["Decision Tree"] = accuracy_score(y_test, y_test_pred)*100
metrics.confusion_matrix(y_test,y_test_pred)
#Use Random Forest classifer
forest_classifier = ensemble.RandomForestClassifier(n_estimators=100)
forest_classifier.fit(X_train, y_train)
forest_y_test_pred = forest_classifier.predict(X_test)
print(metrics.classification_report(y_test, forest_y_test_pred))
print("Accuracy Score :", accuracy_score(y_test, forest_y_test_pred))
Accuracy["Random Forest"] = accuracy_score(y_test, forest_y_test_pred)*100
metrics.confusion_matrix(y_test,forest_y_test_pred)
#Support-Vector Machine
svm_classifier = svm.SVC(gamma="auto")
svm_classifier.fit(X_train, y_train)
svm_y_test_pred = svm_classifier.predict(X_test)
print(metrics.classification_report(y_test, svm_y_test_pred))
print("Accuracy Score :", accuracy_score(y_test, svm_y_test_pred))
Accuracy["SVM"] = accuracy_score(y_test, svm_y_test_pred)*100
metrics.confusion_matrix(y_test,svm_y_test_pred)
#Plot 
df2 = pd.DataFrame.from_dict(Accuracy, orient = 'index', columns = ['Accuracy']) #Create a dataframe based on the accuracy score
plt.figure(figsize = (10,5))
sns.barplot(x=df2.index, y=df2.Accuracy)
plt.xlabel('Classification')
plt.ylabel('Accuracy Score(%)')
plt.title('Accuracy Scores of different Classifications')
plt.show()
