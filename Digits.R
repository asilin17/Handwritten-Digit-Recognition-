########################################################################
# 
# CLASSIFICATION OF HANDWRITTEN ZIP CODE DIGITS USING KNN ALOGRITHM.
# AUTHOR: Alexey Silin
########################################################################

#Importing libraries
library(dplyr)
library(tidyr)
library(ggplot2)

#Data can be downloaded from: https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

#extracting files from zip folder
foldername <- "digits.zip"
unzip(foldername, files = NULL, list = FALSE, overwrite = TRUE, junkpaths = FALSE, exdir =
        "./digits", unzip = "internal", setTimes = FALSE)


######################################################################
# Fuction to read data in
######################################################################

read_digits <- function(filename) {
  
  data <- read.table(filename)
  
  colnames(data)[1] <- "digit"
  data$digit <- factor(data$digit)
  
  data

}
  
training_set <- read_digits("train.txt")
test_set <- read_digits("test.txt")

##################################################################
# Function to display any single digit
#####################################################################


view_digit = function(data, obs) {
  
  a = data[,-1]     #removes 1's column (with digit lables)
  #arranging by column is faster in r
   m = matrix(unlist(a[obs,]), 16,16)  
   
  #function to rotate the matrix
  #https://stackoverflow.com/questions/16496210/rotate-a-matrix-in-r
  #an alternative would be to swap the limits of y-axis when displaying
     rotate = function(x) t(apply(x, 1, rev))   
  
  #displays in a viewer friendly format: 
  #black digit on white background  
  image(rotate((-1)*m),col=paste("gray",1:99,sep=""))           
}

view_digit(training_set, 4)



########################################################################
# Function to predict a digit using knn algorithm
########################################################################
#this function take as argument any vector of points from either training 
#or the test set, then computes the appropriate distance matrix
#only between the selected points and the ones of the training set
#however, for any method, the distance matrix is computed only ones
#Note that the first column of the training set is passed into it as labels

predict_knn = function(points, train, labels, k, method) {
  
  n = nrow(train)
  m = nrow(points)
  
  combined = rbind(train, points)
  mt_combined = as.matrix(combined)
  
  #distance matrix computed within the function
  #but only once and only between the selected test set points and
  #the observations of the training set.
  #User can choose a single point or any combination of points from 
  #any data set, to which they would like to apply the classifier.
  distance_matrix = dist(mt_combined,
                         method = method,
                         diag = T, upper = T)
  
  distance_matrix <- as.matrix(distance_matrix)
  
  
  for (ind in split_indexes) {
    distance_matrix[ind,ind] <- Inf
  }
  
  f = function(col) {
    v = head(order(col), k)
    t = table(labels[c(v)])
    # browser()
    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    result
  }
  
  predictions <- apply(distance_matrix, 2, f)
  names(predictions) <- NULL
  predictions
}

#predicts five nearest neighbors for the first five values of the test set.
#the first column contains the lable for each observation and not a pixel
#value; therefore, it is not used for distance computation. Hence, it's removed.
predict_knn(test_set[1:5,-1], training_set[,-1], training_set[,1], 5, "euclidean")


#########################################################################
# Function to estimate cross-validation errors
#########################################################################
#Here I used slightly different approach than in the section above. 
#unlike in predict_knn(), I did not compute distance matrices within
#the function, since it's two time- and resource-consuming to do it ten times. 
#Instead I wrote a 'helper' function: compute_distances(), which takes as arguments
#a data set for which we running the cross-validation, in this case training, and the desired method
#The cv_error_knn() in turn accepts the resulting distance matrix
#as an argument and performs all the stepps of cross-validation.

set.seed(123) 

#splitting and shuffling the indeces of the training set for 10-fold cross validation
#Here I took an index of each observation from the training set and 
#randomly assigned it into one of the ten equally sized groups (w/o replacement to avoid douplication.)
#Hence, for 10-fold cross-validation in each step I use observations from
#one of the groups as my training segment and the other nine for testing, repeating
#the process 10 times. 

split_indexes <- split(indexes, ceiling(seq_along(indexes)/(n/10)))

#function to compute distance matrix
compute_distances = function(train, method) {
  
  mt_train = as.matrix(train)
  
  distance_matrix = dist(mt_train,
                         method = method,
                         diag = T, upper = T)
  
  distance_matrix <- as.matrix(distance_matrix)
  
  distance_matrix
  
}

#computing three different distance matrices, one for each of the following distance
#metrics: Eucledean, Minkowski, and Manhattan
distance_matrix_euc = compute_distances(training_set[,-1], "euclidean")
distance_matrix_mink = compute_distances(training_set[,-1], "minkowski")
distance_matrix_manh = compute_distances(training_set[,-1], "manhattan")

#function to compute error rates for 10-fold cross validation
cv_error_knn = function(distance_matrix, labels, k) {

  #setting the distance from a data point to itself to infinity for 
  # all the points, since it is always the shortest, but not useful in prediction
  #at all.   
  for (ind in split_indexes) {
    distance_matrix[ind,ind] <- Inf
  }
  
  predictions = apply(distance_matrix, 2, function(col) {
    
    v = head(order(col), k) 
    
    t = table(training_set[v,1])
    # print(t)
    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    factor(result)
  } )
  # browser()
   names(predictions) <- NULL  
   mean(predictions != labels) 
   #using mean function in the statement above is equivalent to summing the incorrect predictoins
   #and dividing by the total number of predictions.

}

#setting the k's for which to cross validate, ex.: 1:15
k = c(1:15)

#pre-allocating memory for the vector of predicted error rates
error_rates_euc = k

#computing error rates for euclidian method
for (i in seq_along(k)) {
  
  error_rates_euc[i] = cv_error_knn(distance_matrix_euc, training_set[,1], i)  
  error_rates_euc[i]
}

#pre-allocating memory for the vector of predicted error rates
error_rates_manh = k

##computing error rates for manhattan method
for (i in seq_along(k)) {
  
  error_rates_manh[i] = cv_error_knn(distance_matrix_manh, training_set[,1], i)  
  error_rates_manh[i]
}


#######################################################
# Plotting 10-fold cv error rates
#######################################################

#combining the results into a dataframe for easy display

errors_rate <- data.frame(k,error_rates_euc, error_rates_manh)

#Note that if I was to perform any analysis on this dataframe
#I would have to tidy it appropriately to make sure that each column 
#is a variabl (here the variables would be: k, method, and error rate 
#See the commented out code section to see how I've done it.

p <- ggplot(errors_rate, aes(k, error_rates_euc, error_rates_manh), axis = T) 

p + geom_line( aes(k, error_rates_euc, col = "euclidean")) + 
  geom_line( aes(k, error_rates_manh, col = "manhattan")) + 
  scale_x_discrete(limits = c(1:15)) + 
  scale_y_continuous(name="Error Rate", limits=c(.02, .075)) + 
  ggtitle("10-Fold CV Error Rates")

######################################################################################
# tidying data frame with error rates for two methods, ex: "eucledean" and "manhattan"
######################################################################################
# colnames(errors_rate) <- c("k", "euclidean", "manhattan")
# error_rates <- 
#   errors_rate %>%
#   gather(method, error_rate, -k) %>%
#   print


#########################################################################
# creating confusion matrices for the three 'best' combinations of k/method 
#########################################################################

#modifying cv_error_knn function to output predictions to input into 
#confusion matrix
cv_predict_knn = function(distance_matrix, labels, k) {
  
  #setting the distance from a point to itself to infinity (see lines 161-163.) 

  for (ind in split_indexes) {
    distance_matrix[ind,ind] <- Inf
  }
  
  predictions = apply(distance_matrix, 2, function(col) {
    
    v = head(order(col), k)
    t = table(training_set[v,1])
     
    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    factor(result)
  } )
  # browser()
  names(predictions) <- NULL
  predictions
  
}

# It looks like the best combinations are in k = 3-5 for 
# both methods explored (euclidean and manhattan). I purposely exclued k= 1 case since it's most likely 
# to show a biased result by comparing a value to itself.
# The three best combinations are [k = 3, euclidean], [k = 3, manhattan*] , [k = 4, euclidean]
# * (the euclidean results are better for # k = 3 : 10, but I've uncluded this one instead to 
# better compare the two methods)

#cv predictions and confusion matrix for [k=3, euclidean]
confusion_euclid_k3 = cv_predict_knn(distance_matrix_euc, training_set[,1], 3)
confusion_matrix_1 <- table(training_set[,1], confusion_euclid_k3)
confusion_matrix_1

#cv predictions and confusion matrix for [k=3, manhattan]
confusion_manhattan_k3 = cv_predict_knn(distance_matrix_manh, training_set[,1], 3)
confusion_matrix_2 <- table(training_set[,1], confusion_manhattan_k3)
confusion_matrix_2

#cv predictions and confusion matrix for [k=4, euclidean]
confusion_euclid_k4 = cv_predict_knn(distance_matrix_euc, training_set[,1], 4)
confusion_matrix_3 <- table(training_set[,1], confusion_euclid_k4)
confusion_matrix_3

##################################################################
# Computing error rates for test set for K = 1,..., 15
##################################################################
#The approach I use here is implementing a hybrid function between
# predict_knn() and cv_error_knn(). Like in the latter, I computed
# the distance matrix beforehand and pass it as an argument into our
#function error_rate_knn(). And like a former, it takes both the training and 
#test data, including the labels for indexing and dimention control, but does
# not perform a cross validation.


#combining test and training set
mt_combined2 = as.matrix(rbind(training_set, test_set))

#computing distance matrix for 
test_set_distance_matrix_euc = compute_distances(mt_combined2[,-1], "euclidean")
test_set_distance_matrix_euc <- as.matrix(test_set_distance_matrix_euc)            #converting distnces into matrix
test_set_distance_matrix_manh = compute_distances(mt_combined2[,-1], "manhattan")
test_set_distance_matrix_manh <- as.matrix(test_set_distance_matrix_manh)          #converting distnces into matrix

#modifying pedict_knn function to return error rates instead of predictions
error_rate_knn = function(points, train, distance_matrix, labels, k, method) {
  
  n = nrow(train)
  m = nrow(points)
  
  #setting the distance from a point to itself to infinity for 
  # all the points, per Ben's OH. 
  for (ind in 1:(n+m)) {
    distance_matrix[ind,ind] <- Inf 
  }
  
  f = function(col) {
    v = head(order(col), k)
    t = table(labels[c(v)])

    result <- names(t)[which(t == max(t))]
    
    if(length(result) > 1)
      result <- sample(result, 1)
    result
  }
  
  predictions <- apply(distance_matrix, 2, f)
  names(predictions) <- NULL
  mean(predictions != labels)
}

test_error_rates_euc = c(1:15)

for (i in seq_along(k)) {
  
  test_error_rates_euc[i] = error_rate_knn(test_set[,-1], training_set[,-1],test_set_distance_matrix_euc, training_set[,1], i)  
  test_error_rates_euc[i]
}

test_error_rates_manh = c(1:15)

for (i in seq_along(k)) {
  
  test_error_rates_manh[i] = error_rate_knn(test_set[,-1], training_set[,-1],test_set_distance_matrix_manh, training_set[,1], i)  
  test_error_rates_manh[i]
}

#combining test set error rates for both methods into a dataframe for 
#easy plotting
test_set_error_rates <- data.frame(k, test_error_rates_euc, test_error_rates_manh)


#Plotting the test set error rates results for k = 1, ..., 15

p <- ggplot(test_set_error_rates, aes(k, test_error_rates_euc, test_error_rates_manh), axis = T) 

p + geom_line( aes(k, test_error_rates_euc, col = "euclidean")) + 
  geom_line( aes(k, test_error_rates_manh, col = "manhattan")) + 
  scale_x_discrete(limits = c(1:15)) + 
  scale_y_continuous(name="Error Rate", limits=c(.2,.4)) + 
  ggtitle("Test Set Error Rate")


###############################################################################
###                             * * *
###############################################################################




 