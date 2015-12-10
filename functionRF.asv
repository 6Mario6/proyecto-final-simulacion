function error = functionRF(Xtrain, Ytrain, Xtest, Ytest)
   
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=normalizar(Xtest,mu,sigma);
         %%%  %%%
    NumArboles=10;
    Modelo=entrenarFOREST(NumArboles,Xtrain,Ytrain);
   
    Yest=testFOREST(Modelo,Xtest);
    Yest = round(Yest);
    error = calculateerrorKN(Ytest, Yest, 0);