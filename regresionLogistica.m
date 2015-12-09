function W = regresionLogistica(X,Y,eta)

[N,D]=size(X);
W = zeros(D,1);
W=W';

for iter = 1:100
    W = W - eta*(1/N)*((sigmoide(W*X'))' - Y)'*X;
end
W=W';
end