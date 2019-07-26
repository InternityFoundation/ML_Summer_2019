import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import random

rootdir = "C:\\Users\\Shridhar\\Desktop\\Enron-Spam"


ham_list = []
spam_list = []

# Same as before, but this time, read the files, and append them to the ham and spam list
for directories, subdirs, files in os.walk(rootdir):
    if (os.path.split(directories)[1]  == 'ham'):
        for filename in files:      
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                ham_list.append(data)
    
    if (os.path.split(directories)[1]  == 'spam'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                spam_list.append(data)


# Write a function , that when passed in words, will return a dictionary of the form

# {Word1: True, Word2: True, Words3: True}

# Removing stop words is optional

def create_word_features(words):
    my_dict = dict( [ (word, True) for word in words] )
    return my_dict

ham_list = []
spam_list = []

# Same as before, but this time:

# 1. Break the sentences into words using word_tokenize
# 2. Use the create_word_features() function you just wrote
for directories, subdirs, files in os.walk(rootdir):
    if (os.path.split(directories)[1]  == 'ham'):
        for filename in files:      
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                
                # The data we read is one big string. We need to break it into words.
                words = word_tokenize(data)
                
                ham_list.append((create_word_features(words), "ham"))
    
    if (os.path.split(directories)[1]  == 'spam'):
        for filename in files:
            with open(os.path.join(directories, filename), encoding="latin-1") as f:
                data = f.read()
                
                # The data we read is one big string. We need to break it into words.
                words = word_tokenize(data)
                
                spam_list.append((create_word_features(words), "spam"))


combined_list = ham_list + spam_list
random.shuffle(combined_list)


# Create a test and train section.
# 70% of the data is training. 30% is test

training_part = int(len(combined_list) * .7)

training_set = combined_list[:training_part]

test_set =  combined_list[training_part:]


# Create the Naive Bayes filter
classifier = NaiveBayesClassifier.train(training_set)

# Find the accuracy, using the test data
accuracy = nltk.classify.util.accuracy(classifier, test_set)

# Clasify the below as spam or ham

# Hint: 1. Break into words using word_tokenzise
# 2. create_word_features
# 3. Use the classify function

msg1 = "Hi I am from Google. You are hired for the job of data scientist."

words = word_tokenize(msg1)
features = create_word_features(words)
print("Message 1 is :" ,classifier.classify(features))
 