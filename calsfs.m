clc
clear
load('DataTest.mat');
    N=size(X,1);
    %N=N*0.001;
    Y(Y<=1000)=1;
    Y(Y>1000 & Y<=2000)=2;
    Y(Y>2000)=3;
   % particion=cvpartition(N,'Kfold',10);
    %X=X(particion.training(10),:);
    %Y=Y(particion.training(10),:);
cvp = cvpartition(Y, 'k', 10);
featuresForY1 = sequentialfs(@functionDG, X, Y, 'cv', cvp);
%featuresForY2 = sequentialfs(@functionRF, X, Y, 'cv', cvp); 
%featuresForY3 = sequentialfs(@functionKN, X, Y, 'cv', cvp);


    