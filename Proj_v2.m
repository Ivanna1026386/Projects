%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Assignment 8 - AEML
Submitted by -
 - Vivek Bhargava (01/1025699)
 - Ivanna Savonik (01/1026386)

No. of functions used -

Other details -


  Data description
  (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
  ------------------------------------------------------------------

  > 

    Number of Instances
    -------------------
    class 1 -> 59
    class 2 -> 71
 	
 
    Number of Attributes - 30
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
indOfClassB = find(strcmp(classes.Var2, 'B')==1);
indOfClassM = setdiff(1:n,indOfClassB)';
X_B = table2array(dataset(indOfClassB,3:k));
X_M = table2array(dataset(indOfClassM,3:k));

rng('default')

Y = strcmp(table2array(classes),'M'); % returns 1 if M 0 if B
% Y = 1 or 'TRUE' ==> Malignant (Positive)
% Y = 0 or 'FALSE' ==> Benign (Negative)
[X_train, X_test, Y_train, Y_test] = splitSample(X_all,Y, 0.8, true);

[X_train_norm, mu_X_train, sigma_X_train] = featureNormalize(X_train);
% Now normalizing testing data using value from training standards
X_test_norm = (X_test - mu_X_train)./sigma_X_train;


% Part 1-a) Training a LDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LDAtrainedClassifier, LDA_Accuracy] = LDAtrainClassifier(X_train_norm, Y_train);
Y_fit_LDA = LDAtrainedClassifier.predictFcn(X_test_norm);
figure
confLDA = confusionchart(Y_test,Y_fit_LDA);

[X1_LDA,Y1_LDA,T1_LDA,AUC1_LDA] = perfcurve(double(Y_test),double(Y_fit_LDA),1)

% Part 1-b) Training a QDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[QDAtrainedClassifier, QDA_Accuracy] = QDAtrainClassifier(X_train_norm, Y_train);
Y_fit_QDA = QDAtrainedClassifier.predictFcn(X_test_norm);
figure
confQDA = confusionchart(Y_test,Y_fit_QDA)

[X1_QDA,Y1_QDA,T1_QDA,AUC1_QDA] = perfcurve(double(Y_test),double(Y_fit_QDA),1)

% Part 1-c) Training a GNB Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[GNBtrainedClassifier, GNB_Accuracy] = GNBtrainClassifier(X_train_norm, Y_train);
Y_fit_GNB = GNBtrainedClassifier.predictFcn(X_test_norm);
figure
confGNB = confusionchart(Y_test,Y_fit_GNB)

[X1_GNB,Y1_GNB,T1_GNB,AUC1_GNB] = perfcurve(double(Y_test),double(Y_fit_GNB),1)


% Part 1-d) Training a Logistic Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LogistictrainedClassifier, Logistic_Accuracy] = LogistictrainClassifier(X_train_norm, Y_train);
Y_fit_Logistic = LogistictrainedClassifier.predictFcn(X_test_norm);
figure
confLogistic = confusionchart(Y_test,Y_fit_Logistic)

[X1_Logistic,Y1_Logistic,T1_Logistic,AUC1_Logistic] = perfcurve(double(Y_test),double(Y_fit_Logistic),1)


% Part 1-e) Training a Linear SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Linear_SVMtrainedClassifier, Linear_SVM_Accuracy] = Linear_SVMtrainClassifier(X_train_norm, Y_train);
Y_fit_Linear_SVM = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
figure
conf_LinearSVM = confusionchart(Y_test,Y_fit_Linear_SVM)

[X1_Linear_SVM,Y1_Linear_SVM,T1_Linear_SVM,AUC1_Linear_SVM] = perfcurve(double(Y_test),double(Y_fit_Linear_SVM),1)


% Part 1-f) Training a Quadratic SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Quadratic_SVMtrainedClassifier, Quadratic_SVM_Accuracy] = Quadratic_SVMtrainClassifier(X_train_norm, Y_train);
Y_fit_Quadratic_SVM = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
figure
confLinearSVM = confusionchart(Y_test,Y_fit_Quadratic_SVM)

[X1_Quadratic_SVM,Y1_Quadratic_SVM,T1_Quadratic_SVM,AUC1_Quadratic_SVM] = perfcurve(double(Y_test),double(Y_fit_Quadratic_SVM),1)

%-----------------------------------------------------------------------
%[X,Y,T,AUC] = perfcurve(double(Y_test),double(Y_fit_Quadratic_SVM),1)
%-----------------------------------------------------------------------

Accuracy = [LDA_Accuracy,QDA_Accuracy,GNB_Accuracy, Logistic_Accuracy,Linear_SVM_Accuracy,Quadratic_SVM_Accuracy]










