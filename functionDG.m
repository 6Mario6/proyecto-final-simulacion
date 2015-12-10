function error = functionDG(Xtrain, Ytrain, Xtest, Ytest)
   
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=normalizar(Xtest,mu,sigma);
         %%%  %%%
    Yest  =classify(Xtest,Xtrain,Ytrain,'linear');
    error = calculateerrorKN(Ytest, Yest, 0);

end

