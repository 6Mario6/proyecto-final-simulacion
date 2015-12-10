clc
clear
load('DataTest.mat');
    N=size(X,1);
    Rept=10;
    N=N*0.003;
    Y(Y<=1000)=1;
    Y(Y>1000 & Y<=2000)=2;
    Y(Y>2000)=3;
    particion=cvpartition(N,'Kfold',3);
    X=X(particion.training(3),:);
    Y=Y(particion.training(3),:);
 

cvp = cvpartition(Y, 'k', 3);
featuresForY1 = sequentialfs(@functionDG, X, Y, 'cv', cvp);
%featuresForY2 = sequentialfs(@functionRF, X, Y, 'cv', cvp); 
%featuresForY3 = sequentialfs(@functionKN, X, Y, 'cv', cvp);   
    