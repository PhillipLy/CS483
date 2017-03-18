
# California State University Fullerton
# CPSC 483: Data Mining and Pattern Recognition - Spring 2017

# Software Engineers: Phillip Ly, Holly Ho, Sara Lipowsky, Tony Dao, James Lindsey
# Courtesy of Stephen Marsland

# Machine learning: Classification of the Heart Disease data set 

def preprocessHeart(infile,outfile):


########################################################
# To do for preprocessing the data start: 
# try to preprocess the original heart-data set ('reprocessed.hungarian') 
# According to the text on page 93, we need to preprocess the data so that later
# on we can evenly split it into training, validation, and test sets
# There is a total of 294 examples in our selected dataset (heart-disease.data).
# We need to place the first 98 examples into training set, the next 98 into validation set, 
# and the last 98 into test set

# For the classification of iris, Stephen did something similar by replace the first 50s (from 'iris-setosa' to '0'
# and 'iris-versicolor to '2', and so on). There were 150 examples total for iris with 0, and the second 50s with 1, etc. 
# 
# We need to code this section to do the above task. If not we will need to manually adjusted it such as 
# place the first 98 with the code word 'train' and then replace it with '0', the next 98 with 'valid' then
# replace it with '1', etc

    stext1 = 'heart-setosa'
    stext2 = 'heart-versicolor'
    stext3 = 'heart-virginica'
    rtext1 = '0'
    rtext2 = '1'

    fid = open(infile,"r")
    oid = open(outfile,"w")
		
    for s in fid:
        if s.find(stext1)>-1:
            oid.write(s.replace(stext1, rtext1))
        elif s.find(stext2)>-1:
            oid.write(s.replace(stext2, rtext2))
        elif s.find(stext3)>-1:
            oid.write(s.replace(stext3, rtext3))
    fid.close()
    oid.close()
# To do ends:
########################################################

import numpy as np
# Preprocessor to remove the test (only needed once)
#preprocessHeart('C:\Heart\Heart\reprocessed.hungarian.data','heart-disease.data')



########################################################
# To do to improve our percentages into the high 80s or 90s starts:

heart = np.loadtxt('heart-disease.data', delimiter=' ')

heart01 = heart[:, 0:1]

heart34 = heart[:, 3:5]

heart9 = heart[:, 9:10]

heart13 = heart[:, 13:]

heart = np.concatenate((heart01, heart34, heart9, heart13), axis=1)

heart[:,:4] = heart[:,:4]-heart[:,:4].mean(axis=0)
imax = np.concatenate((heart.max(axis=0)*np.ones((1,5)),np.abs(heart.min(axis=0)*np.ones((1,5)))),axis=0).max(axis=0)
heart[:,:4] = heart[:,:4]/imax[:4]
print heart[0:5,:]

# Split into training, validation, and test sets
target = np.zeros((np.shape(heart)[0],3));
indices = np.where(heart[:,4]==0) 
target[indices,0] = 1
indices = np.where(heart[:,4]==1)
target[indices,1] = 1
indices = np.where(heart[:,4]==2)
target[indices,2] = 1


# Randomly order the data
order = range(np.shape(heart)[0])
np.random.shuffle(order)
heart = heart[order,:]
target = target[order,:]

train = heart[::2,0:4]
traint = target[::2]
valid = heart[1::4,0:4]
validt = target[1::4]
test = heart[3::4,0:4]
testt = target[3::4]

#print train.max(axis=0), train.min(axis=0)

# Implementation to train the network
import mlp
net = mlp.mlp(train,traint,5,outtype='logistic')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)

# To do ends:
########################################################