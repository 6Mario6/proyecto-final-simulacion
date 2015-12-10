function error = functionKN(Xtrain, Ytrain, Xtest, Ytest)
   
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=normalizar(Xtest,mu,sigma);
         %%%  %%%
    k=10;
    Yest=vecinosCercanos(Xtest,Xtrain,Ytrain,k,'class'); 
    error = calculateerrorKN(Ytest, Yest, 0);