
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


# Processing 'AirQualityUCI.csv' to 'AirQualityUCI.data' (for example) by replacing the 
# ',' in 'AirQualityUCI.csv' with '.' in 'AirQualityUCI.data', etc.
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
		s = s.replace(',', '.')
		s = s.replace(';;', '')
		oid.write(s)


	fid.close()
	oid.close()

	


import pylab as pl
import numpy as np


#Process data once, uncomment to process
preprocessAirData('AirQualityUCI.csv', 'AirQualityUCI.data')


def formatDayTime(data):
    dataRange = np.shape(data)[0]
    dayInMonth = [0,31,59,90,120,151,181,212,243,273,304,334]
	#zero for new data maxtrix of 2 column with same old data rows
    newData = np.zeros((dataRange,2))
	
    #loop each line of old data for convert
    for i in range(dataRange):
        #newdata column 1 = old data last 4 digit for year (Ex: 10032004 => 2004)
        newData[i,0] = (data[i,0] % 10000)
		#get day from old data column1 (Ex: 10032004 => 10)
        day = float(int(data[i,0] / 1000000))
		#get month from old data column1 (Ex: 10032004 => 03)
        month = int(data[i,0] / 10000) % 100
		#newdata column2 = total days in a year (Ex: 10032004 => 10 + 59)
        newData[i,1] = (day + dayInMonth[month-1])
    subData = data[:, 2:]
    #combine the new data with old data except first 2 column
    newData = np.concatenate((newData, subData), axis=1)
    return newData

PNoz = np.loadtxt('AirQualityUCI.data', delimiter=';',skiprows = 1)
#format first 2 columns of data that col1= year, col2=total days in a year
PNoz = formatDayTime(PNoz)
print PNoz

#remove entire row of data that column3 contain -200 or -2000 value
newData = PNoz[0:1,:]
for i in range(np.shape(PNoz)[0]):
    if i != 0:
        if PNoz[i,2] > 0:
            newData = np.append(newData, PNoz[i:i+1,:],axis=0)
PNoz = newData

#check if data contain negative number like -200 or -2000
for i in range(np.shape(PNoz)[0]):
    if PNoz[i,2] < 0:
        print PNoz[i,2]
pl.ion()
#np.arange(3) = [0,1,2]
#np.shape([1,2,3]) = 3 ====> np.shape(PNoz)[0] = 2855 lines
#pl.plot(x,y, optional) x = 0 to 3854, y = third column of data, using point to display in graph
pl.plot(np.arange(np.shape(PNoz)[0]),PNoz[:,2],'.')
pl.xlabel('Time (hours)')
pl.ylabel('CO (in mg/m^3)')

# Normalise data
# take normalize1 = data column 3 - it average
# then normalize2 = nomalize1 / nomalize1 maximum
PNoz[:,2] = PNoz[:,2]-PNoz[:,2].mean()
PNoz[:,2] = PNoz[:,2]/PNoz[:,2].max()

# Assemble input vectors
t = 2
k = 6

#lastPoint = 3855 - 2*(3+1) = 2847
lastPoint = np.shape(PNoz)[0]-t*(k+1)
#input = zero for all 2847x3 (2847 rows and 3 columns)
inputs = np.zeros((lastPoint,k))
#targets = zero for all 2847x1 (2847 rows and 1 column)
targets = np.zeros((lastPoint,1))

for i in range(lastPoint):
	#input row = take only column 2 of PNoz from row i to i+6 skip by 2
    inputs[i,:] = PNoz[i:i+t*k:t,2]
    targets[i] = PNoz[i+t*(k+1),2]

test = inputs[-400:,:]
testtargets = targets[-400:]
train = inputs[:-400:2,:]
traintargets = targets[:-400:2]
valid = inputs[1:-400:2,:]
validtargets = targets[1:-400:2]

# Randomly order the data
# change = [0 to 2846]
change = range(np.shape(inputs)[0])
#randomize the ordered array
np.random.shuffle(change)
inputs = inputs[change,:]
targets = targets[change,:]

# Train the network
import mlp
net = mlp.mlp(train,traintargets,15,outtype='linear')
net.earlystopping(train,traintargets,valid,validtargets,0.25)

test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
testout = net.mlpfwd(test)

pl.figure()
pl.plot(np.arange(np.shape(test)[0]),testout,'.')
pl.plot(np.arange(np.shape(test)[0]),testtargets,'x')
pl.legend(('Predictions','Targets'))
print 0.5*np.sum((testtargets-testout)**2)
pl.show(block = True)
