1. Background
-------------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

2. Desired Results
------------------

The goal of your project is to predict the manner in which they did the
exercise. This is the "classe" variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

#### 2.1. Peer Review Portion

Your submission for the Peer Review portion should consist of a link to
a Github repo with your R markdown and compiled HTML file describing
your analysis. Please constrain the text of the writeup to &lt; 2000
words and the number of figures to be less than 5. It will make it
easier for the graders if you submit a repo with a gh-pages branch so
the HTML page can be viewed online (and you always want to make it easy
on graders :-).

#### 2.2. Course Project Prediction Quiz Portion

Apply your machine learning algorithm to the 20 test cases available in
the test data above and submit your predictions in appropriate format to
the Course Project Prediction Quiz for automated grading.

3. Adding Required Packages
---------------------------

Packages which will be used in this project report are added below. In
case, if any package is not installed then it should be installed before
adding that package to project. And also set seed to make the project
reproducible.

    library(caret)
    library(rpart)
    library(rpart.plot)
    library(corrplot)
    library(randomForest)
    library(RColorBrewer)

    set.seed(123)

4. Fetching Data
----------------

Set the current working directory using setwd(). Then fetch the training
and testing data from provided urls and save them as .csv file.

    setwd("~/Desktop/Coursera")
    trainURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    trainFile <- "./data/training_PML_courseProject.csv"
    testFile  <- "./data/testing_PML_courseProject.csv"
    if (!file.exists(trainFile)) {
      download.file(trainURL, destfile = trainFile, method = "curl")
    }
    if (!file.exists(testFile)) {
      download.file(testURL, destfile = testFile, method = "curl")
    }

5. Processing Data
------------------

#### 5.1. Reading Data

We will read downloaded .csv files in trainData and testData data frames
and explore the dimensions of these data sets. The classe variable in
the trainData and testData data sets is the outcome to be predicted.

    trainData <- read.csv(trainFile)
    testData <- read.csv(testFile)
    dim(trainData)

    ## [1] 19622   160

    dim(testData)

    ## [1]  20 160

#### 5.2. Cleaning Data

Now, we will clean the datasets and eliminate the missing values and
irrelevant or not very contributing parameters.

    #Remove non zero variance variables
    nzv <- nearZeroVar(trainData, saveMetrics = TRUE)
    trainData <- trainData[,!nzv$nzv]
    testData <- testData[,!nzv$nzv]

    #Remove the columns which are not contributing much
    notMuch <- grepl("^X|timestamp|user_name", names(trainData))
    trainData <- trainData[,!notMuch]
    testData <- testData[,!notMuch]

    #Remove NAs
    include <- (colSums(is.na(trainData)) == 0)
    trainData <- trainData[, include]
    testData <- testData[, include]

#### 5.3. Observing Correlation

In order to observe correlation among the parameters of the trainData we
will make a correlation plot.

    corrplot(cor(trainData[, -length(names(trainData))]), method = "circle", tl.cex = 0.5)

![](MachineLearningProject_files/figure-markdown_strict/unnamed-chunk-5-1.png)

#### 5.4. Partitioning trainData Dataset

trainData is partitioned into train and validation data in order to
perform cross validation in upcoming steps.

    inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
    trainn <- trainData[inTrain,]
    validation <- trainData[-inTrain,]

6. Data Modeling
----------------

#### 6.1. Decision Tree

Using decision tree, we will fit a model for prediction purpose.

    modelDT <- rpart(classe~., data = trainn, method = "class")
    predictDT <- predict(modelDT, validation, type = "class")
    confusionMatrix(validation$classe, predictDT)$overall[1]*100

    ## Accuracy 
    ## 83.38148

#### 6.2. Random Forest

We will try Random Forest algorithm because it automatically selects
important variables and is robust to correlated covariates & outliers in
general.

    modelRF <- randomForest(classe~., data = trainn)
    predictRF <- predict(modelRF, validation)
    confusionMatrix(validation$classe, predictRF)$overall[1]*100

    ## Accuracy 
    ## 99.66015

As we can see Random Forest provides higher accuracy of 99.83%, which
was expected.

7. Predicting Manner of Exercise for Test Data
----------------------------------------------

Now, we will apply Random Forest model to the test data downloaded from
the data source to predict the manner of exercise done by subject.
Applying modelRF prediction model to fetch results.

    predict(modelRF, testData[, -length(names(testData))])

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
