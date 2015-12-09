function Modelo = entrenarFOREST(NumArboles,X,Y)
opts = statset('UseParallel',true);
   Modelo = TreeBagger(NumArboles, X, Y,'Options',opts);
    %view(Modelo,'mode','graph')
end