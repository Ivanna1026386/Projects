%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Assignment 8 - AEML
Submitted by -
 - Vivek Bhargava (01/1025699)
 - Ivanna Savonik (01/1026386)

No. of functions used -

Other details -


  Data description (see http://archive.ics.uci.edu/ml/datasets/Wine)
  ------------------------------------------------------------------

  > These data are the results of a chemical analysis of
    wines grown in the same region in Italy but derived from three
    different cultivars.
    The analysis determined the quantities of 13 constituents
    found in each of the three types of wines. 

    Number of Instances
    -------------------
    class 1 -> 59
    class 2 -> 71
 	class 3 -> 48
 
    Number of Attributes - 13
    -------------------------

Classification Lasso

LDA
QDA
Logistic Reg
KNN
GNB
SVM

Model Validation - 
1. ROC - curves
2. Error - 0-1 loss error

Random - forest 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

clear all;
close all;
clc

dataset = readtable('wdbc.csv');
[n,k] = size(dataset);

X_all = table2array(dataset(:,3:k));
classes = (dataset(:, 2));
%
indOfClassB = find(strcmp(classes.Var2, 'B')==1)
indOfClassM = setdiff(1:n,indOfClassB)'
X_B = table2array(dataset(indOfClassB,3:k))
X_M = table2array(dataset(indOfClassM,3:k))



Y = strcmp(table2array(classes),'M') % returns 1 if M 0 if B
% Y = 1 or 'TRUE' ==> Malignant
% Y = 0 or 'FALSE' ==> Benign
[X_train, X_test, Y_train, Y_test] = splitSample(X_all,Y, 0.8, true);

[X_train_norm, mu_X_train, sigma_X_train] = featureNormalize(X_train);
% Now normalizing testing data using value from training standards
X_test_norm = (X_test - mu_X_train)./sigma_X_train;


% Part 1-a) Training a LDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LDAtrainedClassifier, LDA_Accuracy] = LDAtrainClassifier(X_train_norm, Y_train);
Y_fit_LDA = LDAtrainedClassifier.predictFcn(X_test_norm);
figure
conf_LDA = confusionchart(Y_test,Y_fit_LDA)

% Part 1-b) Training a QDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[QDAtrainedClassifier, QDA_Accuracy] = QDAtrainClassifier(X_train_norm, Y_train);
Y_fit_QDA = QDAtrainedClassifier.predictFcn(X_test_norm);
figure
conf_QDA = confusionchart(Y_test,Y_fit_QDA)

% Part 1-c) Training a GNB Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[GNBtrainedClassifier, GNB_Accuracy] = GNBtrainClassifier(X_train_norm, Y_train);
Y_fit_GNB = GNBtrainedClassifier.predictFcn(X_test_norm);
figure
conf_GNB = confusionchart(Y_test,Y_fit_GNB)


% Part 1-d) Training a Logistic Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LogistictrainedClassifier, Logistic_Accuracy] = LogistictrainClassifier(X_train_norm, Y_train);
Y_fit_Logistic = LogistictrainedClassifier.predictFcn(X_test_norm);
figure
conf_Logistic = confusionchart(Y_test,Y_fit_Logistic)


% Part 1-e) Training a Linear SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Linear_SVMtrainedClassifier, Linear_SVM_Accuracy] = Linear_SVMtrainClassifier(X_train_norm, Y_train);
Y_fit_Linear_SVM = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
figure
conf_Linear_SVM = confusionchart(Y_test,Y_fit_Linear_SVM)


% Part 1-f) Training a Quadratic SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Quadratic_SVMtrainedClassifier, Quadratic_SVM_Accuracy] = Quadratic_SVMtrainClassifier(X_train_norm, Y_train);
Y_fit_Quadratic_SVM = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
figure
conf_Quadratic_SVM = confusionchart(Y_test,Y_fit_Quadratic_SVM)















