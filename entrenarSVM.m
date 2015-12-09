function Modelo = entrenarSVM(X,Y,tipo,boxConstraint,sigma,RBF_kernel)

    %%% Completar el codigo %%%
    Modelo = trainlssvm({X,Y,tipo,boxConstraint,sigma,RBF_kernel});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%

end