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

%Y = strcmp(table2array(classes),'M'); % returns 1 if M 0 if B
Y = table2array(dataset(:,2));
% Y = 1 or 'TRUE' ==> Malignant (Positive)
% Y = 0 or 'FALSE' ==> Benign (Negative)
[X_train, X_test, Y_train, Y_test] = splitSample(X_all,Y, 0.8, true);

[X_train_norm, mu_X_train, sigma_X_train] = featureNormalize(X_train);
% Now normalizing testing data using value from training standards
X_test_norm = (X_test - mu_X_train)./sigma_X_train;



%
% Part 1-a) Training a LDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LDAtrainedClassifier, LDA_Accuracy] = LDAtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_LDA_train,score_LDA_train] = LDAtrainedClassifier.predictFcn(X_train_norm);
[X2_LDA,Y2_LDA,T2_LDA,AUC2_LDA] = perfcurve(Y_train,score_LDA_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_LDA,score_LDA] = LDAtrainedClassifier.predictFcn(X_test_norm);
[X1_LDA,Y1_LDA,T1_LDA,AUC1_LDA] = perfcurve(Y_test,score_LDA(:,2),'M');

LDA_Accuracy_test = sum(strcmp(Y_test, Y_fit_LDA))/length(Y_test)
%
%{
% Confusion Chart
figure
confLDA = confusionchart(Y_test,Y_fit_LDA);
%}

% Part 1-b) Training a QDA Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[QDAtrainedClassifier, QDA_Accuracy] = QDAtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_QDA_train,score_QDA_train] = QDAtrainedClassifier.predictFcn(X_train_norm);
[X2_QDA,Y2_QDA,T2_QDA,AUC2_QDA] = perfcurve(Y_train,score_QDA_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_QDA,score_QDA] = QDAtrainedClassifier.predictFcn(X_test_norm);
[X1_QDA,Y1_QDA,T1_QDA,AUC1_QDA] = perfcurve(Y_test,score_QDA(:,2),'M');

QDA_Accuracy_test = sum(strcmp(Y_test, Y_fit_QDA))/length(Y_test)

%{
% Confusion Chart
figure
confQDA = confusionchart(Y_test,Y_fit_QDA)
%}
%
% Part 1-c) Training a GNB Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[GNBtrainedClassifier, GNB_Accuracy] = GNBtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_GNB_train,score_GNB_train]= GNBtrainedClassifier.predictFcn(X_train_norm);
[X2_GNB,Y2_GNB,T2_GNB,AUC2_GNB] = perfcurve(Y_train,score_GNB_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_GNB,score_GNB]= GNBtrainedClassifier.predictFcn(X_test_norm);
[X1_GNB,Y1_GNB,T1_GNB,AUC1_GNB] = perfcurve(Y_test,score_GNB(:,2),'M');

GNB_Accuracy_test = sum(strcmp(Y_test, Y_fit_GNB))/length(Y_test)

%{
% Confusion Chart
figure
confGNB = confusionchart(Y_test,Y_fit_GNB)
%}
%
% Part 1-d) Training a Logistic Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LogistictrainedClassifier, Logistic_Accuracy] = LogistictrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Logistic_train,score_Logistic_train] = LogistictrainedClassifier.predictFcn(X_train_norm);
[X2_Logistic,Y2_Logistic,T2_Logistic,AUC2_Logistic] = perfcurve(Y_train,score_Logistic_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_Logistic,score_Logistic] = LogistictrainedClassifier.predictFcn(X_test_norm);
[X1_Logistic,Y1_Logistic,T1_Logistic,AUC1_Logistic] = perfcurve(Y_test,score_Logistic(:,2),'M');

Logistic_Accuracy_test = sum(strcmp(Y_test, Y_fit_Logistic))/length(Y_test)

%{
% Confusion Chart
figure
confLogistic = confusionchart(Y_test,Y_fit_Logistic)
%}

%
% Part 1-e) Training a Linear SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Linear_SVMtrainedClassifier, Linear_SVM_Accuracy] = Linear_SVMtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Linear_SVM_train, score_Linear_SVM_train] = Linear_SVMtrainedClassifier.predictFcn(X_train_norm);
[X2_Linear_SVM,Y2_Linear_SVM,T2_Linear_SVM,AUC2_Linear_SVM] = perfcurve(Y_train,score_Linear_SVM_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Linear_SVM, score_Linear_SVM] = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
[X1_Linear_SVM,Y1_Linear_SVM,T1_Linear_SVM,AUC1_Linear_SVM] = perfcurve(Y_test,score_Linear_SVM(:,2),'M');

Linear_SVM_Accuracy_test = sum(strcmp(Y_test, Y_fit_Linear_SVM))/length(Y_test)

%{
% Confusion Chart
figure
conf_LinearSVM = confusionchart(Y_test,Y_fit_Linear_SVM)
%}

%
% Part 1-f) Training a Quadratic SVM Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Quadratic_SVMtrainedClassifier, Quadratic_SVM_Accuracy] = Quadratic_SVMtrainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Quadratic_SVM_train,score_Quadratic_SVM_train] = Linear_SVMtrainedClassifier.predictFcn(X_train_norm);
[X2_Quadratic_SVM,Y2_Quadratic_SVM,T2_Quadratic_SVM,AUC2_Quadratic_SVM] = perfcurve(Y_train,score_Quadratic_SVM_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Quadratic_SVM,score_Quadratic_SVM] = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
[X1_Quadratic_SVM,Y1_Quadratic_SVM,T1_Quadratic_SVM,AUC1_Quadratic_SVM] = perfcurve(Y_test,score_Quadratic_SVM(:,2),'M');

Quadratic_SVM_Accuracy_test = sum(strcmp(Y_test, Y_fit_Quadratic_SVM))/length(Y_test)

% Confusion Chart
%{
figure
confLinearSVM = confusionchart(Y_test,Y_fit_Quadratic_SVM)
%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comapiring Results with all 30 features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compairing Accuracy
Accuracy_Training = [LDA_Accuracy,QDA_Accuracy,GNB_Accuracy, Logistic_Accuracy,Linear_SVM_Accuracy,Quadratic_SVM_Accuracy]
Accuracy_Testing = [LDA_Accuracy_test, QDA_Accuracy_test, GNB_Accuracy_test, Logistic_Accuracy_test, Linear_SVM_Accuracy_test, Quadratic_SVM_Accuracy_test]

% Compairing Area Under the Curve
AUC2_Training_data = [AUC2_LDA, AUC2_QDA, AUC2_GNB, AUC2_Logistic, AUC2_Linear_SVM, AUC2_Quadratic_SVM]
AUC1_Testing_data = [AUC1_LDA, AUC1_QDA, AUC1_GNB, AUC1_Logistic, AUC1_Linear_SVM, AUC1_Quadratic_SVM]



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part 2 - Feature Selection and Model Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Part 2-a) Training a LDA Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LDAtrainedClassifier, LDA_Accuracy] = LDA_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_LDA_train,score_LDA_train] = LDAtrainedClassifier.predictFcn(X_train_norm);
[X2_LDA,Y2_LDA,T2_LDA,AUC2_LDA] = perfcurve(Y_train,score_LDA_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_LDA,score_LDA] = LDAtrainedClassifier.predictFcn(X_test_norm);
[X1_LDA,Y1_LDA,T1_LDA,AUC1_LDA] = perfcurve(Y_test,score_LDA(:,2),'M');

LDA_Accuracy_test = sum(strcmp(Y_test, Y_fit_LDA))/length(Y_test)



% Part 2-b) Training a QDA Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[QDAtrainedClassifier, QDA_Accuracy] = QDA_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_QDA_train,score_QDA_train] = QDAtrainedClassifier.predictFcn(X_train_norm);
[X2_QDA,Y2_QDA,T2_QDA,AUC2_QDA] = perfcurve(Y_train,score_QDA_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_QDA,score_QDA] = QDAtrainedClassifier.predictFcn(X_test_norm);
[X1_QDA,Y1_QDA,T1_QDA,AUC1_QDA] = perfcurve(Y_test,score_QDA(:,2),'M');

QDA_Accuracy_test = sum(strcmp(Y_test, Y_fit_QDA))/length(Y_test)



% Part 2-c) Training a GNB Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[GNBtrainedClassifier, GNB_Accuracy] = GNB_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_GNB_train,score_GNB_train]= GNBtrainedClassifier.predictFcn(X_train_norm);
[X2_GNB,Y2_GNB,T2_GNB,AUC2_GNB] = perfcurve(Y_train,score_GNB_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_GNB,score_GNB]= GNBtrainedClassifier.predictFcn(X_test_norm);
[X1_GNB,Y1_GNB,T1_GNB,AUC1_GNB] = perfcurve(Y_test,score_GNB(:,2),'M');

GNB_Accuracy_test = sum(strcmp(Y_test, Y_fit_GNB))/length(Y_test)



% Part 2-d) Training a Logistic Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LogistictrainedClassifier, Logistic_Accuracy] = Logistic_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Logistic_train,score_Logistic_train] = LogistictrainedClassifier.predictFcn(X_train_norm);
[X2_Logistic,Y2_Logistic,T2_Logistic,AUC2_Logistic] = perfcurve(Y_train,score_Logistic_train(:,2),'M');

% Calculation AUC and Accuracy for testing data
[Y_fit_Logistic,score_Logistic] = LogistictrainedClassifier.predictFcn(X_test_norm);
[X1_Logistic,Y1_Logistic,T1_Logistic,AUC1_Logistic] = perfcurve(Y_test,score_Logistic(:,2),'M');

Logistic_Accuracy_test = sum(strcmp(Y_test, Y_fit_Logistic))/length(Y_test)



% Part 2-e) Training a Linear SVM Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Linear_SVMtrainedClassifier, Linear_SVM_Accuracy] = Linear_SVM_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Linear_SVM_train, score_Linear_SVM_train] = Linear_SVMtrainedClassifier.predictFcn(X_train_norm);
[X2_Linear_SVM,Y2_Linear_SVM,T2_Linear_SVM,AUC2_Linear_SVM] = perfcurve(Y_train,score_Linear_SVM_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Linear_SVM, score_Linear_SVM] = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
[X1_Linear_SVM,Y1_Linear_SVM,T1_Linear_SVM,AUC1_Linear_SVM] = perfcurve(Y_test,score_Linear_SVM(:,2),'M');

Linear_SVM_Accuracy_test = sum(strcmp(Y_test, Y_fit_Linear_SVM))/length(Y_test)



% Part 2-f) Training a Quadratic SVM Model Using PCA - 95% (variance explained)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Quadratic_SVMtrainedClassifier, Quadratic_SVM_Accuracy] = Quadratic_SVM_PCA_trainClassifier(X_train_norm, Y_train);

% Calculating AUC for trained data
[Y_fit_Quadratic_SVM_train,score_Quadratic_SVM_train] = Linear_SVMtrainedClassifier.predictFcn(X_train_norm);
[X2_Quadratic_SVM,Y2_Quadratic_SVM,T2_Quadratic_SVM,AUC2_Quadratic_SVM] = perfcurve(Y_train,score_Quadratic_SVM_train(:,2),'M');


% Calculation AUC and Accuracy for testing data
[Y_fit_Quadratic_SVM,score_Quadratic_SVM] = Linear_SVMtrainedClassifier.predictFcn(X_test_norm);
[X1_Quadratic_SVM,Y1_Quadratic_SVM,T1_Quadratic_SVM,AUC1_Quadratic_SVM] = perfcurve(Y_test,score_Quadratic_SVM(:,2),'M');

Quadratic_SVM_Accuracy_test = sum(strcmp(Y_test, Y_fit_Quadratic_SVM))/length(Y_test)












%}
%}
%}
%}
%}



%{
[idx,scores] = fscchi2(X_train_norm,Y_train);

tree=fitctree(X_train_norm,Y_train);


view(tree,'Mode','graph');



imp=predictorImportance(tree);
[r,ind_imp]=sort(imp, 'descend');
index_zero=find(imp==0);



data_train =[Y_train, X_train_norm];
Y_pred_tree=predict(tree, X_test_norm);
confusionchart(Y_test, Y_pred_tree);
C=confusionmat(Y_test,Y_pred_tree);
%Accuracy=(TN+TP)/(TN+TP+FN+FP)
accuracy_ctree=(C(1,1)+C(2,2))/length(Y_test);
%}
%}





