
# California State University Fullerton
# CPSC 483: Data Mining and Pattern Recognition - Spring 2017

# Software Engineers: Phillip Ly, Holly Ho, Sara Lipowsky, Tony Dao, James Lindsey
# Courtesy of Stephen Marsland

# Machine learning: Regression of the Air Quality Dataset 


# for preprocessing the Air quality data from 'AirQualityUCI.csv' (source file)
# and 'AirQualityUCI.data' (processed file), we have to play around with regular expression
# to get pass errors from noise in the original dataset such 2,6 
# helpful stackoverflow:  http://stackoverflow.com/questions/16720541/python-string-replace-regular-expression

# As of now, I have managed to get by all the errors but currently our graphs are not looking like
# time-series as illustrated in the book, we will have to improve the preprocessing of 
# AirQualityUCI.csv to get more accurate looking graphs

# Another alternative is to just use another dataset, I am also working on the alternative/backup 
# dataset to do regression on, I am also working on the wine dataset (UCI's repo) as our backup just in case

#from datetime import datetime

def preprocessAirData(infile,outfile):

	fid = open(infile, 'r')
	oid = open(outfile, 'w')
	firstLine = True
	for s in fid:
		if firstLine:
			firstLine = False
			oid.write(s)
			continue			
		s = s.replace('/', '')
		s = s.replace('.00.00', '')
		s = s.replace(',', '')
		s = s.replace(';;', '')
		oid.write(s)


	fid.close()
	oid.close()

	
#What I was trying to do: http://stackoverflow.com/questions/36427418/plotting-a-date-from-an-csv-file-in-pylab
	
import pylab as pl
import numpy as np
#import csv

#Process data once, uncomment to process
preprocessAirData('AirQualityUCI.csv', 'AirQualityUCI.data')

#Air = np.genfromtxt('AirQualityUCI.csv', names=True, usecols=(0, 1, 2, 3), delimiter= ';', dtype =[('Date', 'S10'),('Time', '<f8'), ('CO(GT)', '<f8'), ('PT08.S1', '<f8')])

Air = np.loadtxt('AirQualityUCI.data', delimiter=';',skiprows = 1)
pl.ion()
pl.plot(np.arange(np.shape(Air)[0]),Air[:,2],'.')
pl.xlabel('Date')
pl.ylabel('CO (in mg/m^3)')


#x =  [foo['Date'] for foo in Air]

#dates=[]
#for i in x:
#    date = datetime.strptime(i,'%d/%m/%Y')
#    dates.append(date)


# Normalise data
Air[:,2] = Air[:,2]-Air[:,2].mean()
Air[:,2] = Air[:,2]/Air[:,2].max()


# Suppose that t = 2 and k = 3. Then the first input data are elements 1, 3, 5 
# of the dataset, and the target is element 7. The next input vector is elements 
# 2, 4, 6, with target 8, and then 3, 5, 7 with target 9. You train the network
# by passing through the time-series (remembering to save some data for testing), and then
# press on into the future making predictions.


# Assemble input vectors
t = 2
k = 6

lastPoint = np.shape(Air)[0]-t*(k+1)
inputs = np.zeros((lastPoint,k))
targets = np.zeros((lastPoint,1))
for i in range(lastPoint):
    inputs[i,:] = Air[i:i+t*k:t,2]
    targets[i] = Air[i+t*(k+1),2]

# There is a total of 9358 samples of data 
# We are going to split it up to roughly 3000
# each for test, train, and validation sets
    
test = inputs[-3000:,:]
testtargets = targets[-3000:]
train = inputs[:-3000:2,:]
traintargets = targets[:-3000:2]
valid = inputs[1:-3000:2,:]
validtargets = targets[1:-3000:2]

# Randomly order the data
change = range(np.shape(inputs)[0])
np.random.shuffle(change)
inputs = inputs[change,:]
targets = targets[change,:]

# Train the network
import mlp
net = mlp.mlp(train,traintargets,3,outtype='linear')
net.earlystopping(train,traintargets,valid,validtargets,0.25)

test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
testout = net.mlpfwd(test)

pl.figure()
pl.plot(np.arange(np.shape(test)[0]),testout,'.')
pl.plot(np.arange(np.shape(test)[0]),testtargets,'x')
pl.legend(('Predictions','Targets'))
print 0.5*np.sum((testtargets-testout)**2)
pl.show(block = True)
