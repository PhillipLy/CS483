
# California State University Fullerton
# CPSC 483: Data Mining and Pattern Recognition - Spring 2017

# Software Engineers: Phillip Ly, Holly Ho, Sara Lipowsky, Tony Dao, James Lindsey
# Courtesy of Stephen Marsland

# Machine learning: Classification of the Heart Disease data set 


import numpy as np

########################################################
# To do to improve our percentages into the high 80s or 90s starts:

#load the data, then grad the values we want
heart = np.loadtxt('reprocessed.hungarian.data', delimiter=' ')

heart0 = heart[:, 0:1]  # Get age from the data set

heart1 = heart[:, 1:2] #Sex

heart2 = heart[:, 2:3] #chest pain type

heart34 = heart[:,3:5]  # resting blood pressure and serum cholestoral in mg/dl

heart5 = heart[:,5:6] #fasting blood sugar

heart6 = heart[:,6:7] # resting electrocardiographic results

# The new element I added is below
heart7 = heart[:, 7:8]  # thalach: maximum heart rate achieved

heart8 = heart[:, 8:9] # exang: exercise induced angina (1 = yes; 0 = no)

heart9 = heart[:, 9:10]  # oldpeak = ST depression induced by exercise relative to rest

heart13 = heart[:, 13:]  # Get target

# heart2, heart 8, heart9 and heart13 are the current best
heart = np.concatenate((heart1, heart8, heart9, heart13), axis=1)

# these two numbers must be changed. nont is the number of non-target columns
# t is the index of the target column
nont = 3
t = 4
heart[:,:nont] = heart[:,:nont]-heart[:,:nont].mean(axis=0)
imax = np.concatenate((heart.max(axis=0)*np.ones((1,t)),np.abs(heart.min(axis=0)*np.ones((1,t)))),axis=0).max(axis=0)
heart[:,:nont] = heart[:,:nont]/imax[:nont]
print heart[0:5,:]

# Split into training, validation, and test sets

# note: changes here because we only have two possible outputs: 0 and 1
# as such, we only need a 1xN array
target = np.zeros((np.shape(heart)[0],1));
indices = np.where(heart[:,nont]>=1)
target[indices,0] = 1

# Randomly order the data
order = range(np.shape(heart)[0])
np.random.shuffle(order)
# Note, the values below give an accuracy of 93.15-94.5%
# order = [160, 49, 26, 234, 264, 270, 182, 170, 193, 124, 28, 113, 82, 189, 201, 162, 101, 252, 57, 292, 81, 53, 87, 103, 110, 78, 172, 44, 72, 167, 251, 121, 56, 255, 10, 52, 174, 9, 131, 69, 126, 30, 239, 268, 147, 51, 266, 195, 276, 97, 137, 215, 205, 15, 108, 212, 169, 39, 120, 34, 3, 213, 105, 197, 73, 289, 125, 190, 161, 2, 272, 98, 74, 265, 31, 154, 181, 175, 123, 285, 37, 80, 129, 38, 60, 198, 258, 284, 13, 256, 76, 164, 106, 75, 238, 185, 48, 153, 150, 209, 259, 116, 71, 157, 85, 83, 241, 24, 1, 40, 58, 191, 217, 54, 135, 122, 27, 257, 290, 199, 47, 178, 180, 282, 93, 222, 17, 70, 102, 152, 254, 23, 220, 288, 263, 19, 231, 267, 95, 274, 260, 207, 62, 155, 43, 32, 244, 130, 11, 145, 63, 240, 96, 20, 293, 141, 41, 118, 262, 188, 166, 200, 286, 202, 236, 187, 55, 0, 67, 219, 214, 242, 133, 142, 235, 208, 12, 25, 291, 211, 277, 112, 114, 94, 192, 144, 250, 117, 90, 176, 61, 158, 84, 186, 86, 247, 221, 206, 269, 99, 246, 109, 115, 146, 50, 65, 139, 132, 168, 163, 88, 184, 280, 218, 36, 6, 119, 171, 42, 287, 29, 179, 228, 224, 136, 134, 46, 229, 261, 127, 138, 92, 140, 203, 271, 232, 14, 177, 100, 148, 18, 8, 104, 226, 216, 21, 156, 281, 128, 5, 173, 111, 151, 149, 143, 7, 165, 16, 230, 243, 253, 91, 249, 4, 204, 278, 196, 248, 273, 245, 35, 107, 279, 45, 89, 227, 223, 59, 283, 183, 210, 79, 77, 68, 275, 233, 33, 22, 237, 66, 225, 194, 64, 159]
heart = heart[order,:]
target = target[order,:]

#need to change train, valid and test too
train = heart[::2,0:nont]
traint = target[::2]
valid = heart[1::4,0:nont]
validt = target[1::4]
test = heart[3::4,0:nont]
testt = target[3::4]


# Implementation to train the network
import mlp
net = mlp.mlp(train,traint,3,outtype='logistic')
net.earlystopping(train,traint,valid,validt,0.1)
# page 92
# net.confmat(valid, validt)
net.confmat(test,testt)
