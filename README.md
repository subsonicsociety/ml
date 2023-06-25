1. Implement and demonstrate the FIND-S algorithm for finding the most specific 
hypothesis based on a given set of training data samples. Read the training data from a 
.CSV file.
Find-s Algorithm :
1. Load Data set
2. Initialize h to the most specific hypothesis in H
3. For each positive training instance x
• For each attribute constraint ai in h
If the constraint ai in h is satisfied by x then do nothing
else replace ai in h by the next more general constraint that is satisfied by x
4. Output hypothesis h
   
Source Code:
import random
import csv
def read_data(filename):
 with open(filename, 'r') as csvfile:
 datareader = csv.reader(csvfile, delimiter=',')
 traindata = []
 for row in datareader:
 traindata.append(row)
 return (traindata)
h=['phi','phi','phi','phi','phi','phi'
data=read_data('finds.csv')
def isConsistent(h,d):
 if len(h)!=len(d)-1:
 print('Number of attributes are not same in hypothesis.')
 return False
 else:
 matched=0 
 for i in range(len(h)):
 if ( (h[i]==d[i]) | (h[i]=='any') ): 
 matched=matched+1
 if matched==len(h):
 return True
 else:
 return False
def makeConsistent(h,d):
for i in range(len(h)):
 if((h[i] == 'phi')):
 h[i]=d[i]
 elif(h[i]!=d[i]):
 h[i]='any'
 return h
print('Begin : Hypothesis :',h)
15CSL76 ML LAB
Dept. of CSE,MSEC Page 8
print('==========================================')
for d in data:
 if d[len(d)-1]=='Yes':
 if ( isConsistent(h,d)): 
 pass
 else:
 h=makeConsistent(h,d)
 print ('Training data :',d)
 print ('Updated Hypothesis :',h)
 print()
 print('--------------------------------')
print('==========================================')
print('maximally sepcific data set End: Hypothesis :',h)

Output:
Begin : Hypothesis : ['phi', 'phi', 'phi', 'phi', 'phi', 'phi']
==========================================
Training data : ['Cloudy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'Yes']
Updated Hypothesis : ['Cloudy', 'Cold', 'High', 'Strong', 'Warm', 'Change']
--------------------------------
Training data : ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
Updated Hypothesis : ['any', 'any', 'any', 'Strong', 'Warm', 'any']
--------------------------------
Training data : ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
Updated Hypothesis : ['any', 'any', 'any', 'Strong', 'Warm', 'any']
--------------------------------
Training data : ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
Updated Hypothesis : ['any', 'any', 'any', 'Strong', 'any', 'any']
--------------------------------
Training data : ['Overcast', 'Cool', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
Updated Hypothesis : ['any', 'any', 'any', 'Strong', 'any', 'any']
--------------------------------
==========================================
maximally sepcific data set End: Hypothesis : ['any', 'any', 'any', 'Strong', 'any', 'any']

OR

import csv
def loadCsv(filename):
 lines = csv.reader(open(filename, "r"))
 dataset = list(lines)
 for i in range(len(dataset)):
 dataset[i] = dataset[i]
 return dataset
attributes = ['Sky','Temp','Humidity','Wind','Water','Forecast']
print('Attributes =',attributes)
num_attributes = len(attributes)
filename = "finds.csv"
dataset = loadCsv(filename)
print(dataset)
hypothesis=['0'] * num_attributes
print("Intial Hypothesis")
print(hypothesis)
print("The Hypothesis are")
for i in range(len(dataset)):
 target = dataset[i][-1]
 if(target == 'Yes'):
 for j in range(num_attributes):
 if(hypothesis[j]=='0'):
 hypothesis[j] = dataset[i][j]
 if(hypothesis[j]!= dataset[i][j]):
 hypothesis[j]='?'
 print(i+1,'=',hypothesis)
print("Final Hypothesis")
print(hypothesis)

Output:
Attributes = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast']
[['sky', 'Airtemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'WaterSport'], 
['Cloudy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'Yes'], 
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'], 
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'], 
['Cloudy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'], 
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes'], 
['Rain', 'Mild', 'High', 'Weak', 'Cool', 'Change', 'No'], 
['Rain', 'Cool', 'Normal', 'Weak', 'Cool', 'Same', 'No'], 
['Overcast', 'Cool', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']]
Intial Hypothesis
['0', '0', '0', '0', '0', '0']
The Hypothesis are
2 = ['Cloudy', 'Cold', 'High', 'Strong', 'Warm', 'Change']
3 = ['?', '?', '?', 'Strong', 'Warm', '?']
4 = ['?', '?', '?', 'Strong', 'Warm', '?']
6 = ['?', '?', '?', 'Strong', '?', '?']
9 = ['?', '?', '?', 'Strong', '?', '?']
Final Hypothesis
['?', '?', '?', 'Strong', '?', '?']



2. For a given set of training data examples stored in a .CSV file, implement and 
demonstrate the Candidate-Elimination algorithm to output a description of the set of all 
hypotheses consistent with the training examples.
Candidate-Elimination Algorithm:
1. Load data set
2. G <-maximally general hypotheses in H 
3. S <- maximally specific hypotheses in H 
4. For each training example d=<x,c(x)> 
Case 1 : If d is a positive example 
Remove from G any hypothesis that is inconsistent with d 
For each hypothesis s in S that is not consistent with d 
• Remove s from S. 
• Add to S all minimal generalizations h of s such that
• h consistent with d 
• Some member of G is more general than h 
• Remove from S any hypothesis that is more general than another hypothesis in S 
Case 2: If d is a negative example 
Remove from S any hypothesis that is inconsistent with d 
For each hypothesis g in G that is not consistent with d 
*Remove g from G. 
*Add to G all minimal specializations h of g such that 
o h consistent with d 
o Some member of S is more specific than h 
• Remove from G any hypothesis that is less general than another hypothesis in G 

Source Code:
import numpy as np
import pandas as pd
data = pd.DataFrame(data=pd.read_csv('finds1.csv'))
concepts = np.array(data.iloc[:,0:-1])
target = np.array(data.iloc[:,-1])
def learn(concepts, target):
 specific_h = concepts[0].copy()
 print("initialization of specific_h and general_h")
 print(specific_h)
 general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
 print(general_h)
 for i, h in enumerate(concepts):
 if target[i] == "Yes":
 for x in range(len(specific_h)):
 if h[x] != specific_h[x]:
 specific_h[x] = '?'
 general_h[x][x] = '?'
 if target[i] == "No":
 for x in range(len(specific_h)):
 if h[x] != specific_h[x]:
 general_h[x][x] = specific_h[x]
 else:
 general_h[x][x] = '?'
 print(" steps of Candidate Elimination Algorithm",i+1)
 print("Specific_h ",i+1,"\n ")
 print(specific_h)
 print("general_h ", i+1, "\n ")
 print(general_h)
 indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
 for i in indices:
 general_h.remove(['?', '?', '?', '?', '?', '?'])
  return specific_h, general_h
s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")

OUTPUT
initialization of specific_h and general_h
['Cloudy' 'Cold' 'High' 'Strong' 'Warm' 'Change']
[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', 
'?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
steps of Candidate Elimination Algorithm 8
Specific_h 8 
['?' '?' '?' 'Strong' '?' '?']
general_h 8 
[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', 'Strong', '?', '?'], ['?', 
'?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
Final Specific_h:
['?' '?' '?' 'Strong' '?' '?']
Final General_h:
[['?', '?', '?', 'Strong', '?', '?']]



3. Write a program to demonstrate the working of the decision tree based ID3 algorithm. 
Use an appropriate data set for building the decision tree and apply this knowledge to 
classify a new sample.

Source Code:
import numpy as np
import math
from data_loader import read_data
class Node:
 def __init__(self, attribute):
 self.attribute = attribute
 self.children = []
 self.answer = ""
  def __str__(self):
 return self.attribute
def subtables(data, col, delete):
 dict = {}
 items = np.unique(data[:, col])
 count = np.zeros((items.shape[0], 1), dtype=np.int32) 
  for x in range(items.shape[0]):
 for y in range(data.shape[0]):
 if data[y, col] == items[x]:
 count[x] += 1 
 for x in range(items.shape[0]):
 dict[items[x]] = np.empty((int(count[x]), data.shape[1]), dtype="|S32")
 pos = 0
 for y in range(data.shape[0]):
 if data[y, col] == items[x]:
 dict[items[x]][pos] = data[y]
 pos += 1 
 if delete:
 dict[items[x]] = np.delete(dict[items[x]], col, 1) 
 return items, dict  
def entropy(S):
 items = np.unique(S)
 if items.size == 1:
 return 0 
 counts = np.zeros((items.shape[0], 1))
 sums = 0 
 for x in range(items.shape[0]):
 counts[x] = sum(S == items[x]) / (S.size * 1.0)
 for count in counts:
 sums += -1 * count * math.log(count, 2)
 return sums 
def gain_ratio(data, col):
 items, dict = subtables(data, col, delete=False) 
 total_size = data.shape[0]
 entropies = np.zeros((items.shape[0], 1))
 intrinsic = np.zeros((items.shape[0], 1)) 
 for x in range(items.shape[0]):
 ratio = dict[items[x]].shape[0]/(total_size * 1.0)
 entropies[x] = ratio * entropy(dict[items[x]][:, -1])
 intrinsic[x] = ratio * math.log(ratio, 2) 
 total_entropy = entropy(data[:, -1])
 iv = -1 * sum(intrinsic)
 for x in range(entropies.shape[0]):
 total_entropy -= entropies[x]
 return total_entropy / iv
def create_node(data, metadata):
 #TODO: Co jeśli information gain jest zerowe?
 if (np.unique(data[:, -1])).shape[0] == 1:
 node = Node("")
 node.answer = np.unique(data[:, -1])[0]
 return node
 gains = np.zeros((data.shape[1] - 1, 1)) 
 for col in range(data.shape[1] - 1):
 gains[col] = gain_ratio(data, col)
 split = np.argmax(gains)
 node = Node(metadata[split]) 
 metadata = np.delete(metadata, split, 0) 
 items, dict = subtables(data, split, delete=True)
 for x in range(items.shape[0]):
 child = create_node(dict[items[x]], metadata)
 node.children.append((items[x], child))
 return node 
def empty(size):
 s = ""
 for x in range(size):
 s += " "
 return s
def print_tree(node, level):
 if node.answer != "":
 print(empty(level), node.answer)
 return
 print(empty(level), node.attribute)
 for value, n in node.children:
 print(empty(level + 1), value)
 print_tree(n, level + 2)
metadata, traindata = read_data("tennis.data")
data = np.array(traindata)
node = create_node(data, metadata)
print_tree(node, 0)

OUTPUT:
outlook
 overcast
 b'yes'
 rain
15CSL76 ML LAB
Dept. of CSE,MSEC Page 15
 wind
 b'strong'
 b'no'
 b'weak'
 b'yes'
 sunny
 humidity
 b'high'
 b'no'
 b'normal'
 b'yes'
 
OR

import pandas as pd
import numpy as np
dataset= pd.read_csv('playtennis.csv',names=['outlook','temperature','humidity','wind','class',])
def entropy(target_col):
 elements,counts = np.unique(target_col,return_counts = True)
 entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in 
range(len(elements))])
 return entropy
def InfoGain(data,split_attribute_name,target_name="class"):
 total_entropy = entropy(data[target_name])
 vals,counts= np.unique(data[split_attribute_name],return_counts=True)
 Weighted_Entropy = 
np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dr
opna()[target_name]) for i in range(len(vals))])
 Information_Gain = total_entropy - Weighted_Entropy
 return Information_Gain 
def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None): 
 if len(np.unique(data[target_attribute_name])) <= 1:
 return np.unique(data[target_attribute_name])[0]
 elif len(data)==0:
 return 
np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribut
e_name],return_counts=True)[1])]
 elif len(features) ==0:
 return parent_node_class
 else:
 parent_node_class = 
np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return
_counts=True)[1])]
 item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return 
the information gain values for the features in the dataset
 best_feature_index = np.argmax(item_values)
 best_feature = features[best_feature_index]
 tree = {best_feature:{}}
 features = [i for i in features if i != best_feature]
 for value in np.unique(data[best_feature]):
 value = value
 sub_data = data.where(data[best_feature] == value).dropna()
 subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
 tree[best_feature][value] = subtree
 return(tree) 
tree = ID3(dataset,dataset,dataset.columns[:-1])
print(' \nDisplay Tree\n',tree)

OUTPUT:
Display Tree
{'outlook': {'Overcast': 'Yes', 'Rain': {'wind': {'Strong': 'No', 'Weak': 'Yes'}}, 'Sunny': 
{'humidity': {'High': 'No', 'Normal': 'Yes'}}}}



4. Build an Artificial Neural Network by implementing the Back propagation Algorithm 
and test the same using appropriate data sets.

Source Code:
import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) # maximum of X array longitudinally y = y/100
#Sigmoid Function
def sigmoid (x):
 return (1/(1 + np.exp(-x)))
#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
 return x * (1 - x)
#Variable initialization
epoch=7000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons)) 
# draws a random range of numbers uniformly of dim x*y
#Forward Propagation
for i in range(epoch):
 hinp1=np.dot(X,wh)
 hinp=hinp1 + bh
 hlayer_act = sigmoid(hinp)
 outinp1=np.dot(hlayer_act,wout)
 outinp= outinp1+ bout
 output = sigmoid(outinp)
#Backpropagation
 EO = y-output
 outgrad = derivatives_sigmoid(output)
 d_output = EO* outgrad
 EH = d_output.dot(wout.T)
 hiddengrad = derivatives_sigmoid(hlayer_act)
#how much hidden layer wts contributed to error
 d_hiddenlayer = EH * hiddengrad
 wout += hlayer_act.T.dot(d_output) *lr
# dotproduct of nextlayererror and currentlayerop
 bout += np.sum(d_output, axis=0,keepdims=True) *lr 
 wh += X.T.dot(d_hiddenlayer) *lr
#bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)

Output:
Input:
[[ 0.66666667 1. ]
[ 0.33333333 0.55555556]
[ 1. 0.66666667]]

Actual Output:
[[ 0.92]
[ 0.86]
[ 0.89]]
Predicted Output:
[[ 0.89559591]
[ 0.88142069]
[ 0.8928407 ]]



5. Write a program to implement the naïve Bayesian classifier for a sample training data 
set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data 
sets.

Source Code:
import csv
import random
import math
def loadCsv(filename):
lines = csv.reader(open(filename, "r"))
dataset = list(lines)
for i in range(len(dataset)):
dataset[i] = [float(x) for x in dataset[i]]
return dataset
def splitDataset(dataset, splitRatio):
trainSize = int(len(dataset) * splitRatio)
trainSet = []
copy = list(dataset)
while len(trainSet) < trainSize:
index = random.randrange(len(copy))
trainSet.append(copy.pop(index))
return [trainSet, copy]
def separateByClass(dataset):
separated = {}
for i in range(len(dataset)):
vector = dataset[i]
if (vector[-1] not in separated):
separated[vector[-1]] = []
separated[vector[-1]].append(vector)
return separated
def mean(numbers):
return sum(numbers)/float(len(numbers))
def stdev(numbers):
avg = mean(numbers)
variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
return math.sqrt(variance)
def summarize(dataset):
summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
del summaries[-1]
return summaries
def summarizeByClass(dataset):
separated = separateByClass(dataset)
summaries = {}
for classValue, instances in separated.items():
summaries[classValue] = summarize(instances)
return summaries
def calculateProbability(x, mean, stdev):
exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
def calculateClassProbabilities(summaries, inputVector):
probabilities = {}
for classValue, classSummaries in summaries.items():
probabilities[classValue] = 1
for i in range(len(classSummaries)):
mean, stdev = classSummaries[i]
x = inputVector[i]
probabilities[classValue] *= calculateProbability(x, mean, stdev)
return probabilities
def predict(summaries, inputVector):
probabilities = calculateClassProbabilities(summaries, inputVector)
bestLabel, bestProb = None, -1
for classValue, probability in probabilities.items():
if bestLabel is None or probability > bestProb:
bestProb = probability
bestLabel = classValue
return bestLabel
def getPredictions(summaries, testSet):
predictions = []
for i in range(len(testSet)):
result = predict(summaries, testSet[i])
predictions.append(result)
return predictions
def getAccuracy(testSet, predictions):
correct = 0
for i in range(len(testSet)):
if testSet[i][-1] == predictions[i]:
correct += 1
return (correct/float(len(testSet))) * 100.0
def main():
filename = 'data.csv'
splitRatio = 0.67
dataset = loadCsv(filename)
trainingSet, testSet = splitDataset(dataset, splitRatio)
print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), 
len(trainingSet), len(testSet)))
# prepare model
summaries = summarizeByClass(trainingSet)
# test model
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}%'.format(accuracy))
main()

OUTPUT :
Split 306 rows into train=205 and test=101 rows
Accuracy: 72.27722772277228%



6. Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier 
model to perform this task. Built-in Java classes/API can be used to write the program. 
Calculate the accuracy, precision, and recall for your data set.

Source Code:
import pandas as pd
msg=pd.read_csv('naivetext1.csv',names=['message','label'])
print('The dimensions of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
print(X)
print(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y)
print(xtest.shape)
print(xtrain.shape)
print(ytest.shape)
print(ytrain.shape) 
 from sklearn.feature_extraction.text import CountVectorizer 
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain) 
predicted = clf.predict(xtest_dtm)
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Recall and Precison ')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))

Output:
The dimensions of the dataset (18, 2)
0 I love this sandwich
1 This is an amazing place
2 I feel very good about these beers
3 This is my best work
4 What an awesome view
5 I do not like this restaurant
6 I am tired of this stuff
7 I can't deal with this
8 He is my sworn enemy
9 My boss is horrible
10 This is an awesome place
11 I do not like the taste of this juice
12 I love to dance
13 I am sick and tired of this place
14 What a great holiday
15 That is a bad locality to stay
16 We will have good fun tomorrow
17 I went to my enemy's house today
Name: message, dtype: object
0 1
1 1
2 1
3 1
4 1
5 0
6 0
7 0
8 0
9 0
10 1
11 0
12 1
13 0
14 1
15 0
16 1
17 0
Name: labelnum, dtype: int64
(5,)
(13,)
(5,)
(13,)
Accuracy metrics
Accuracy of the classifer is 0.8
Confusion matrix
[[3 1]
[0 1]]
Recall and Precison 
1.0
0.5



7. Write a program to construct a Bayesian network considering medical data. Use this 
model to demonstrate the diagnosis of heart patients using standard Heart Disease 
Data Set. You can use Java/Python ML library classes/API.

Source Code:
import numpy as np
from urllib.request import urlopen
import urllib
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 
'thal', 'heartdisease'] 
heartDisease = pd.read_csv('heart.csv', names = names)
heartDisease = heartDisease.replace('?', np.nan) 
model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('exang', 
'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'), 
('heartdisease','thalach'), ('heartdisease','chol')])
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 37, 'sex' :0})
print(q['heartdisease'])

OUTPUT:
╒════════════════╤════
│ heartdisease │ phi(heartdisease) │
╞══════════════════════
│ heartdisease_0 │ 0.5593 │
├─────────────────────┤
│ heartdisease_1 │ 0.4407 │
╘════════════════╧═════



8. Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set 
for clustering using k-Means algorithm. Compare the results of these two algorithms and 
comment on the quality of clustering. You can add Java/Python ML library classes/API in 
the program.

Source Code:
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
X=pd.read_csv("kmeansdata.csv")
x1 = X['Distance_Feature'].values
x2 = X['Speeding_Feature'].values
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
plt.plot()
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()
#code for EM
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
em_predictions = gmm.predict(X) 
print("\nEM predictions")
print(em_predictions) 
print("mean:\n",gmm.means_)
print('\n')
print("Covariances\n",gmm.covariances_)
print(X)
plt.title('Exceptation Maximum')
plt.scatter(X[:,0], X[:,1],c=em_predictions,s=50)
plt.show()
#code for Kmeans
import matplotlib.pyplot as plt1
kmeans = KMeans(n_clusters=3) 
kmeans.fit(X) 
print(kmeans.cluster_centers_) 
print(kmeans.labels_) 
plt.title('KMEANS') 
plt1.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow') 
plt1.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black') 

OUTPUT:
EM predictions
[0 0 0 1 0 1 1 1 2 1 2 2 1 1 2 1 2 1 0 1 0 1 1]
mean:
[[57.70629058 25.73574491]
[52.12044022 22.46250453]
[46.4364858 39.43288647]]
Covariances
[[[83.51878796 14.926902 ]
 [14.926902 2.70846907]]
[[29.95910352 15.83416554]
 [15.83416554 67.01175729]]
[[79.34811849 29.55835938]
 [29.55835938 18.17157304]]]
[[71.24 28. ]
[52.53 25. ]
[64.54 27. ]
[55.69 22. ]
[54.58 25. ]
[41.91 10. ]
[58.64 20. ]
[52.02 8. ]
[31.25 34. ]
[44.31 19. ]
[49.35 40. ]
[58.07 45. ]
[44.22 22. ]
[55.73 19. ]
[46.63 43. ]
[52.97 32. ]
[46.25 35. ]
[51.55 27. ]
[57.05 26. ]
[58.45 30. ]
[43.42 23. ]
[55.68 37. ]
[55.15 18. ]
centroid and predications
[[57.74090909 24.27272727]
[48.6 38. ]
[45.176 16.4 ]]
[0 0 0 0 0 2 0 2 1 2 1 1 2 0 1 1 1 0 0 0 2 1 0]



9. Write a program to implement k-Nearest Neighbour algorithm to classify the iris data 
set. Print both correct and wrong predictions. Java/Python ML library classes can be used 
for this problem.

Source Code:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
dataset=pd.read_csv("iris.csv") 
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.25) 
classifier=KNeighborsClassifier(n_neighbors=8,p=3,metric='euclidean')
classifier.fit(X_train,y_train)
 #predict the test resuts
y_pred=classifier.predict(X_test)
 cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix is as follows\n',cm)
print('Accuracy Metrics')
print(classification_report(y_test,y_pred)) 
print(" correct predicition",accuracy_score(y_test,y_pred))
print(" worng predicition",(1-accuracy_score(y_test,y_pred)))

Output :
Confusion matrix is as follows
[[13 0 0]
[ 0 15 1]
[ 0 0 9]]
Accuracy Metrics
 precision recall f1-score support
 Iris-setosa 1.00 1.00 1.00 13
Iris-versicolor 1.00 0.94 0.97 16
Iris-virginica 0.90 1.00 0.95 9
 avg / total 0.98 0.97 0.97 38
correct predicition 0.9736842105263158
worng predicition 0.02631578947368418



10. Implement the non-parametric Locally Weighted Regression algorithm in order to fit 
data points. Select appropriate data set for your experiment and draw graphs.

Source Code:
import numpy as np
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.io import push_notebook
def local_regression(x0, X, Y, tau):
# add bias term
x0 = np.r_[1, x0] # Add one to avoid the loss in information
X = np.c_[np.ones(len(X)), X]
# fit model: normal equations with kernel
xw = X.T * radial_kernel(x0, X, tau) # XTranspose * W
beta = np.linalg.pinv(xw @ X) @ xw @ Y # @ Matrix Multiplication or Dot Product
# predict value
return x0 @ beta # @ Matrix Multiplication or Dot Product for prediction
def radial_kernel(x0, X, tau):
return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))
# Weight or Radial Kernal Bias Function
n = 1000
# generate dataset
X = np.linspace(-3, 3, num=n)
print("The Data Set ( 10 Samples) X :\n",X[1:10])
Y = np.log(np.abs(X ** 2 - 1) + .5)
print("The Fitting Curve Data Set (10 Samples) Y :\n",Y[1:10])
# jitter X
X += np.random.normal(scale=.1, size=n)
print("Normalised (10 Samples) X :\n",X[1:10])
domain = np.linspace(-3, 3, num=300)
print(" Xo Domain Space(10 Samples) :\n",domain[1:10])
def plot_lwr(tau):
# prediction through regression
prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
plot = figure(plot_width=400, plot_height=400)
plot.title.text='tau=%g' % tau
plot.scatter(X, Y, alpha=.3)
plot.line(domain, prediction, line_width=2, color='red')
return plot
# Plotting the curves with different tau
show(gridplot([
[plot_lwr(10.), plot_lwr(1.)],
[plot_lwr(0.1), plot_lwr(0.01)]
]))

Output:
The Data Set ( 10 Samples) X :
[-2.99399399 -2.98798799 -2.98198198 -2.97597598 -2.96996997 -2.96396396
-2.95795796 -2.95195195 -2.94594595]
The Fitting Curve Data Set (10 Samples) Y :
[2.13582188 2.13156806 2.12730467 2.12303166 2.11874898 2.11445659
2.11015444 2.10584249 2.10152068]
Normalised (10 Samples) X :
[-3.10518137 -3.00247603 -2.9388515 -2.79373602 -2.84946247 -2.85313888
-2.9622708 -3.09679502 -2.69778859]
Xo Domain Space(10 Samples) :
[-2.97993311 -2.95986622 -2.93979933 -2.91973244 -2.89966555 -2.87959866
-2.85953177 -2.83946488 -2.81939799]

OR

from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg 
from scipy.stats.stats import pearsonr
def kernel(point,xmat, k):
 m,n = shape(xmat)
 weights = mat(eye((m)))
 for j in range(m):
 diff = point - X[j]
 weights[j,j] = exp(diff*diff.T/(-2.0*k**2))
 return weights
def localWeight(point,xmat,ymat,k):
 wei = kernel(point,xmat,k)
 W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
 return W
 def localWeightRegression(xmat,ymat,k):
 m,n = shape(xmat)
 ypred = zeros(m)
 for i in range(m):
 ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
 return ypred
# load data points
data = pd.read_csv('tips.csv')
bill = array(data.total_bill)
tip = array(data.tip)
#preparing and add 1 in bill
mbill = mat(bill)
mtip = mat(tip)
m= shape(mbill)[1]
one = mat(ones(m))
X= hstack((one.T,mbill.T))
#set k here
ypred = localWeightRegression(X,mtip,0.2)
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]
 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='green')
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show();

Output:
Dataset 
Add Tips.csv (256 rows)
