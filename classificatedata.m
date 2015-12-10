function value = classificatedata(X, Y)
    classes = max(Y);
    for i = 1:classes
       indexes = (Y == i);
       value{i} = X(indexes, :);
    end