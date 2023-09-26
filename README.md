# MBTI-Project
In this project, I am using the dataset of people with different MBTIs' last 50 comments to conduct a general data analysis of the most frequently used words of each MBTI.
## 1. Data Exploration
After importing the libraries and the dataset (from Kaggle) needed to conduct my analysis, I moved to the data exploration process of my original data to get to know more about my data. I first determine whether there is a null value in the dataset, how many columns and rows from the table, if all the MBTIs are included in the dataset, and the number of posts from each MBTI (which I also generated a graph for better visualization). This helps me to know more about my dataset which enables me to do further data cleaning.
## 2. Data Cleaning
Moving on to the next stage, which is data cleaning. I believe that it was necessary to remove things such as the "|" which is used to separate each comment of a specific person, links, symbols, and words that were too long and too short. After all the removal, I tried to make every word lowercase, since I am finding the wordcloud of the most frequently used words. I also removed MBTI words (e.g. INFP) from the comments and the stopwords before the next process, because this are meaningless words that people used to write sentences with.
## 3. Data Visualization
I used WordCloud to conduct the data visualization process. I created graphs of the most frequently used words among all the people from the dataset and among the 12 MBTIs respectively.
## 4. Simple Machine Learning Process
I used scikit-learn library to determine the accuracy scores of the 3 classifications - Decision Tree, Random Forest Classification, and SVM. It helps me to determine which classification has the most correct predictions from the model.
