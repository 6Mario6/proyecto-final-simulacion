function dis = DistanciaEuclidiana(Xtrain, Xval)
    N = size(Xtrain, 1);
    dis = zeros(N, 1);

    for i = 1:N
        temp = (Xtrain(i, :) - Xval) .^ 2;
        dis(i) = sqrt(sum(temp, 2));
    end