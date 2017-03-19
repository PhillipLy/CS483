
# California State University Fullerton
# CPSC 483: Data Mining and Pattern Recognition - Spring 2017

# Software Engineers: Phillip Ly, Holly Ho, Sara Lipowsky, Tony Dao, James Lindsey
# Courtesy of Stephen Marsland

# Machine learning: Classification of the Heart Disease data set 

def preprocessHeart(infile,outfile):


########################################################

# This function doesn't do anything right now, previously it preprocessed the iris dataset
# If this isn't removed, we should actually make it preprocess the heart dataset



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
    fid.close()
    oid.close()
# To do ends:
########################################################

import numpy as np
# Preprocessor to remove the test (only needed once)
#preprocessHeart('C:\Heart\Heart\reprocessed.hungarian.data','heart-disease.data')



########################################################
# To do to improve our percentages into the high 80s or 90s starts:

#load the data, then grad the values we want
heart = np.loadtxt('heart-disease.data', delimiter=' ')

heart01 = heart[:, 0:1]  # Get age from the data set

heart34 = heart[:, 3:5]  # resting blood pressure and serum cholestoral in mg/dl

# The new element I added is below
heart7 = heart[:, 7:8]  # thalach: maximum heart rate achieved

heart9 = heart[:, 9:10]  # oldpeak = ST depression induced by exercise relative to rest

heart13 = heart[:, 13:]  # Get target

# heart7 was added into the concatenate, between heart34 and heart9
heart = np.concatenate((heart01, heart34, heart7, heart9, heart13), axis=1)

print heart[0:5,:]

# from lines 68-70, all 4's change to 5's, and any 5's change to 6's
heart[:,:5] = heart[:,:5]-heart[:,:5].mean(axis=0)
imax = np.concatenate((heart.max(axis=0)*np.ones((1,6)),np.abs(heart.min(axis=0)*np.ones((1,6)))),axis=0).max(axis=0)
heart[:,:5] = heart[:,:5]/imax[:5]
print heart[0:5,:]

# Split into training, validation, and test sets

# note: changes here because we only have two possible outputs: 0 and 1
# as such, we only need a 2x1 array
target = np.zeros((np.shape(heart)[0],1));
#indices = np.where(heart[:,4]==0)
#target[indices,0] = 1
indices = np.where(heart[:,4]==1)
target[indices,0] = 1


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