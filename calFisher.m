clear;
clc;

load('DataTest.mat');
Y(Y<=1000)=1;
Y(Y>1000 & Y<=2000)=2;
Y(Y>2000)=3;
dataForClases = classificatedata(X, Y);

fishers = calculatefisher(dataForClases);

