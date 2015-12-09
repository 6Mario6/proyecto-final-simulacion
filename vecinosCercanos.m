function f = vecinosCercanos(Xval,Xent,Yent,k,tipo)

    %%% El parametro 'tipo' es el tipo de modelo de que se va a entrenar %%%

    N=size(Xent,1);
    M=size(Xval,1);
    
    f=zeros(M,1);
   % dis=zeros(N,1);
    if strcmp(tipo,'class')
        
        for j=1:M
            dis = DistanciaE(Xent, Xval(j, :));
            [dis, sortIndexes] = sort(dis);
            temp = Yent(sortIndexes(1 : k));
           %%% Complete el codigo %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            f(j)=mode(temp(1:k)); 
        end 
    elseif strcmp(tipo,'regress')
        for j=1:M
            
            dis = DistanciaE(Xent, Xval(j, :));
            [dis, sortIndexes] = sort(dis);
            temp = Yent(sortIndexes(1 : k));
           
            %%% Complete el codigo %%%
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            f(j)=mean(temp(1:k));
            
        end
        
    end

end