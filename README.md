# Heart Disease and Air Quality Datasets
**CPSC 483 - Data Mining and Pattern Recognition - Implementation of Machine Learning Algorithms** 

**Team: Phillip Ly, Holly Ho, Sara Lipowsky, Tony Dao, James Lindsey**

**Link to report and PPT slides: https://goo.gl/ab6WQy**

The goal is to train, validate, and test Multi-Layer Perceptrons or RBF Networks by implementing 
classification and regression (including time series prediction over a period of 8000 hours) algorithms 
on Heart Disease and Air Quality data sets from UC Irvine Machine Learning Repository. We've attained 
an accuracy as high as 95% for classification of heart disease.

**Steps to Replicate Our Results**

**Prerequisites:**
- Python 2.7
- numPy
- matplotLib

1) Download project from GitHub repository:
- In terminal type: git clone https://github.com/PhillipLy/CS483.git
- Notice there are 2 folders- Heart and Air. 

**For Classification:**
2) Change into “Heart” directory:
- cd Heart
- python heart.py
- The range of % that we got
- The range is 73% - 94.5%
- Changes to the code from the original
- Used a different dataset (called reprocessed.hungarian.data)
- Created matrices that allowed access of all usable features
- Modified the test matrix to fit the output of our algorithm (only 0 or 1 instead of 0-2)
- Made global variables so that it is easier to test different features
- Changed the number of hidden nodes from 5 to 3

**For Regression**
3) Change into “Air” directory
- cd Air
- python air.py
- The range of % that we got
- The Error Range: 22% - 26.67% with t = 3 and k = 10
- Changes to the code from the original
- Used reprocess data (called AirQualityUCI.data) and skip the first line of data
- Create a method to remove rows that contain -200 or -2000 in column 3
- Change the plot x and y label
- Change t and k value 2 and 3 to 3 and 10 
- Changed the number of hidden nodes from 3 to 4
- Changed the learning rate from 0.25 to 0.3

