# Zip Code Digit Recognition Project

## Author: Alexey Silin

### Overview:
#### The U.S. Postal Service processes an average of 2.1 million pieces of mail per hour. Outbound mail needs to be sorted according to the zip code of its destination. In the recent years USPS has been adopting the automated mail sorting. The sorting machineds use machine learning classifiers to identify the individual digits in the zip code on each piece of mail. 
#### In this project I've built a suprvised machine learning classifier based on KNN algorithm to classify handwritten digits in zip codes. Zip codes only contain digits from 0 to 9; however, similar approach can be used to classify any known handwritten symbols regardless of a particular language or representationa system. 

### Data: 
#### The data for this project has been taken from the UC Irvine Machine Learning repository: https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
#### Each observation in the dataset represents a handwritten image of a singe digit from 0 to 9 and contains 256 features, which correspond to an individual pixel on a 16x16 plane. Each pixel value ranges from -1 to 1 to indicate the coloration on the grayscale. 

#### Here is an expample of how a single digit looks like, zero in this case:  Handwritten-Digit-Recognition-/Images/0.Rplot.pdf 
#### In order to display it in a more familiar black-on-white, we can multiply all the pixel values by -1, and get the opposite color values for each pixel, like this:  Handwritten-Digit-Recognition-/Images/Zero.Rplot.png 

### Results: 
#### Below is the confusion matrix for K = 3 and distance computation method (i.e. distance metric)
#### The values on the top indicate the observations for which the predictions were made. The values on the left-hand side show which digits our obsrvation(s) in questing have been classified as. If we look at the first column, which says '6', we can see that in 4 instances, it has been classified as 0, once as 2, 6 times as 4, etc. The amoun of correct predictions, wher '6' has been actually identified as such, is 649. In the same way, we can see the results for other digits. Some digits get confused with certain others more often than with the rest. If we look at 9' in 8th column, we can see that ninteen times it has been misclassified as '4'. That's because 4 and 9 can look very similar the way some people write them. You can find to be the case in most instancses with other digits, as well. 
####  Handwritten-Digit-Recognition-/Results/Confusion Matrix_k3_euclid.png 

#### 





