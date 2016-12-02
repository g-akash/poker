from sklearn import preprocessing
import numpy as np
import unicodecsv
import csv


trainfile="train.csv"
testfile="test1.csv"

learning_rate=0.1

c=[]
t=[]
fout=[]
l=[1,0,0,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,1,0,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,1,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,1,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,1,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,1,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,1,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,1,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,1,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,1,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,1,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,0,1,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,0,0,1]
c.append(l)

l=[1,0,0,0]
t.append(l)
l=[0,1,0,0]
t.append(l)
l=[0,0,1,0]
t.append(l)
l=[0,0,0,1]
t.append(l)


l=[1,0,0,0,0,0,0,0,0,0]
fout.append(l)
l=[0,1,0,0,0,0,0,0,0,0]
fout.append(l)
l=[0,0,1,0,0,0,0,0,0,0]
fout.append(l)
l=[0,0,0,1,0,0,0,0,0,0]
fout.append(l)
l=[0,0,0,0,1,0,0,0,0,0]
fout.append(l)
l=[0,0,0,0,0,1,0,0,0,0]
fout.append(l)
l=[0,0,0,0,0,0,1,0,0,0]
fout.append(l)
l=[0,0,0,0,0,0,0,1,0,0]
fout.append(l)
l=[0,0,0,0,0,0,0,0,1,0]
fout.append(l)
l=[0,0,0,0,0,0,0,0,0,1]
fout.append(l)



def derivative(x):
	#return 1.0-x*x
	return (x)*(1.0-(x))

def sigmoid(x):
	#return np.tanh(x)
	return 1.0/(1.0+np.exp(-x))


#make training data here
def filereader(file,tt):
    with open(file,'rb') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        #might need to pop the header
        data_list.pop(0)
        for i in range(len(data_list)):
            if(tt):
                data_list[i].pop(0)
            #might need to pop the index
            for j in range(len(data_list[i])):
                data_list[i][j]=int(data_list[i][j])

        return data_list

def sanitize(data_list):
    outputs=list()
    for i in range(len(data_list)):
        outputs.append((data_list[i][-1]))
        data_list[i].pop()

    return data_list,outputs

def convert_input(x):
	ans=[]
	#print x,"printing x"
	ans.extend(t[x[0]-1])
	ans.extend(c[x[1]-1])
	ans.extend(t[x[2]-1])
	ans.extend(c[x[3]-1])
	ans.extend(t[x[4]-1])
	ans.extend(c[x[5]-1])
	ans.extend(t[x[6]-1])
	ans.extend(c[x[7]-1])
	ans.extend(t[x[8]-1])
	ans.extend(c[x[9]-1])
	return ans


def convert_inputs(input):
	for i in range(len(input)):
		input[i]=convert_input(input[i])
	return input

def convert_outputs(output):
	for i in range(len(output)):
		output[i]=fout[output[i]]
	return output

def convert_back(output):
	for i in range(len(output)):
		for j in range(10):
			if(output[i][j]>.5):
				output[i][j]=1
			else:
				output[i][j]=0
		for j in range(10):
			if (output[i]==np.array(fout[j])).all():
				output[i]=[j]
				break
	return output





train_data=filereader(trainfile,False)
X,y=sanitize(train_data)


#X=X/20.0
#X = preprocessing.scale(X) # feature scaling

#y=y/10.0

#X=X[0:20]
#y=y[0:20]

#print y
X=X[0:120000]
y=y[0:120000]
X=convert_inputs(X)
y=convert_outputs(y)

#print type(y),type(y[0]),type(y[0][0])
#print X,y

for i in range(len(X)):
    X[i]=np.array(X[i])
    y[i]=np.array(y[i])
X=np.array(X)
y=np.array(y)
#make test data here

test_input=filereader(testfile,True)
#print "read file"

#test_input = test_input/20.0
#test_input=preprocessing.scale(test_input)
#test_input = X

#preprocess the data here

test_input = convert_inputs(test_input)

for i in range(len(test_input)):
    test_input[i]=np.array(test_input[i])
#print "made array"

test_input=np.array(test_input)


#test_input=X


dim1 = len(X[0])
dim2 = 18
dim3 = 10
dim4 = 10


np.random.seed(1)

weight0 = 2*np.random.random((dim1,dim2))-1
weight1 = 2*np.random.random((dim2,dim3))-1
weight2 = 2*np.random.random((dim3,dim4))-1



#print len(weight1)
for j in range(200):
 	#print j
	for k in range(len(X)):

	 	layer_0 = np.array([X[k]])
		layer_1 = sigmoid(np.dot(layer_0,weight0))
		layer_2 = sigmoid(np.dot(layer_1,weight1))
		layer_3 = sigmoid(np.dot(layer_2,weight2))
		
		
		#print len(y),len(y[0]),len(layer_2),len(layer_2[0])
		
		
		layer_3_error = np.array([y[k]]) - layer_3
		#print layer_3_error
		#print len(layer_2_error),len(layer_2_error[0])
		layer_3_delta = layer_3_error * derivative(layer_3)



		#print len(layer_2_delta),len(layer_2_delta[0]),len(weight1.T),len(weight1.T[0])
		layer_2_error = layer_3_delta.dot(weight2.T)
		layer_2_delta = layer_2_error * derivative(layer_2)

		layer_1_error = layer_2_delta.dot(weight1.T)
		layer_1_delta = layer_1_error * derivative(layer_1)

		
		
		weight2 += learning_rate*layer_2.T.dot(layer_3_delta)
		weight1 += learning_rate*layer_1.T.dot(layer_2_delta)
		weight0 += learning_rate*layer_0.T.dot(layer_1_delta)

layer_0 = test_input
layer_1 = sigmoid(np.dot(layer_0,weight0))
layer_2 = sigmoid(np.dot(layer_1,weight1))
layer_3 = sigmoid(np.dot(layer_2,weight2))




print "id,CLASS"
layer_3=convert_back(layer_3)
for x in range(len(layer_3)):
	#print X[x],([y[x]])
	print ",".join([str(int(x)),str(int(layer_3[x][0]))])
#er= y-layer_7
#print np.sum(er)