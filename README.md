# Zip Code Digit Recognition Project

## Author: Alexey Silin

### Overview:
#### The U.S. Postal Service processes an average of 2.1 million pieces of mail per hour. Outbound mail needs to be sorted according to the zip code of its destination. In the recent years USPS has been adopting the automated mail sorting. The sorting machineds use machine learning classifiers to identify the individual digits in the zip code on each piece of mail. 
#### In this project I've built a suprvised machine learning classifier based on KNN algorithm to classify handwritten digits in zip codes. Zip codes only contain digits from 0 to 9; however, similar approach can be used to classify any known handwritten symbols regardless of a particular language or representationa system. 

### Data: 
#### The data for this project has been taken from the UC Irvine Machine Learning repository: https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
#### Each observation in the dataset represents a handwritten image of a singe digit from 0 to 9 and contains 256 features, which correspond to an individual pixel on a 16x16 plane. Each pixel value ranges from -1 to 1 to indicate the coloration on the grayscale. 

#### Here is an expample of how a single digit looks like: 


