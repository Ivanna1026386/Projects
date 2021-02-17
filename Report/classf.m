function err = classf(xtrain,ytrain,xtest,ytest)
        
%CLASSF Summary of this function goes here
%   Detailed explanation goes here

yfit = classify(xtest,xtrain,ytrain,'quadratic');
err = sum(~strcmp(ytest,yfit));

end

