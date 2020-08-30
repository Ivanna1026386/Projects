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


Mdl = fitcdiscr(X_train_norm, Y_train,'DiscrimType','Quadratic')
label = predict(Mdl,X_train_norm)

sum(~strcmp(Y_fit_QDA_train, label))


[h,p,ci,stat] = ttest2(X_M,X_B,'Vartype','unequal');
[ZZZZZ,featureIdxSortbyP] = sort(p,2);

nfs = 1:1:30
for i = 1:length(nfs)
   fs = featureIdxSortbyP(1:nfs(i));
   Mdl = fitcdiscr(X_train_norm(:,fs), Y_train,'DiscrimType','Quadratic');
   label = predict(Mdl,X_train_norm(:,fs));
   QDA_Misclassification(:,i)= sum(~strcmp(Y_train, label))/length(Y_train); % Resubs error
   
   
   label2 = predict(Mdl,X_test_norm(:,fs));
   QDA_Misclassification_testingdata(:,i) = sum(~strcmp(Y_test, label2))/length(Y_test);
   
end

QDA_Misclassification
QDA_Misclassification_testingdata

figure
plot(1:length(nfs),QDA_Misclassification)
hold on
plot(1:length(nfs),QDA_Misclassification_testingdata)
xlabel('Number of Features');
ylabel('MCE')
legend({'MCE on the training set' 'MCE on the test set'});
title('Misclassification  - Training and Testing data')
hold off


ZZ = heatmap(corr(X_train_norm))


tenfoldCVP = cvpartition(Y_train,'kfold',10)

fun = @(XT,yT,Xt,yt)loss(fitcdiscr(XT,yT,'DiscrimType','Quadratic'),Xt,yt)
opts = statset('Display','iter');
[fsLocal, history] = sequentialfs(fun,X_train_norm,Y_train,'cv',tenfoldCVP,'options',opts)


X_train_norm(:,fsLocal)

Mdl1 = fitcdiscr(X_train_norm(:,fsLocal), Y_train,'DiscrimType','Quadratic');
label1 = predict(Mdl1,X_test_norm(:,fsLocal));
QDA_Misclassification1= sum(strcmp(Y_test, label1))/length(Y_test); % Resubs error

fs1 = featureIdxSortbyP(1:30)

[fsCVfor30,historyCV] = sequentialfs(fun,X_train_norm,Y_train,...
    'cv',tenfoldCVP,'Nf',30);
plot(historyCV.Crit,'o');
xlabel('Number of Features');
ylabel('CV MCE');
title('Forward Sequential Feature Selection with cross-validation');




holdoutCVP = cvpartition(Y_train,'holdout',0.2)
resubCVP = cvpartition(length(Y_train),'resubstitution')
classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));

for i = 1:length(nfs)
   fs = featureIdxSortbyP(1:nfs(i));
  testMCE(i) = crossval(classf,X_train_norm(:,fs),Y_train,'partition',holdoutCVP)...
       /holdoutCVP.TestSize;
   
   
   resubMCE(i) = crossval(classf,X_train_norm(:,fs),Y_train,'partition',resubCVP)/...
       resubCVP.TestSize;
   
end

 plot(nfs, testMCE,'o',nfs,resubMCE,'r^');
 xlabel('Number of Features');
 ylabel('MCE');
 legend({'MCE on the test set' 'Resubstitution MCE'},'location','NW');
 title('Simple Filter Feature Selection Method');


 %Hyper parameter Optimization
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hypopts = struct('ShowPlots',false,'Verbose',0) %,'UseParallel',true);

mdls_SVM_opt = fitcsvm(X_train_norm,Y_train,'KernelFunction','polynomial','Standardize','on', ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
results_SVM_opt = mdls_SVM_opt.HyperparameterOptimizationResults;


mdls_tree_opt = fitctree(X_train_norm,Y_train, ...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions', hypopts);
results_tree_opt = mdls_tree_opt.HyperparameterOptimizationResults;


figure
plot(results_SVM_opt.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
hold on
plot(results_tree_opt.ObjectiveMinimumTrace,'Marker','o','MarkerSize',5);
names = {'SVM-Polynomial','Decision Tree'};
legend(names,'Location','northeast')
title('Bayesian Optimization')
xlabel('Number of Iterations')
ylabel('Minimum Objective Value')

% Checking it on Testing Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[label_SVM_opt,score_SVM_opt] = predict(mdls_SVM_opt,X_test_norm);

[label_tree_opt,score_tree_opt] = predict(mdls_tree_opt,X_test_norm);


figure
subplot(1,2,1)
c_svm_opt = confusionchart(Y_test,label_SVM_opt);
title(names{1})
subplot(1,2,2)
c_tree_opt = confusionchart(Y_test,label_tree_opt);
title(names{2})





