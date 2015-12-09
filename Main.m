
clc
close all
clear all

eta=0.1; %%% Taza de aprendizaje
grado=1;     %%% Grado del polinomio

punto=input('Ingrese, 1 para Classification Discriminant, 2 para k vecinos Cercanos, 3 para Redes Neuronales, 4 para Random Forest, 5 para  SVM ');

if punto==1
    
    %%%Discriminante gaussiano%%%
    fprintf('Discriminante gaussiano')
    %%% Se crean los datos de forma aleatoria %%%
    load('DataTest.mat');
    N=size(X,1);
    Rept=10;
    N=N*0.3;
  
    Y(Y<=1000)=1;
    Y(Y>1000 & Y<=2000)=2;
    Y(Y>2000)=3;
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
    EficienciaTest=zeros(1,Rept);
 
    unqY=unique(Y);
    countElY=histc(Y,unqY);
    relFreqY=countElY/numel(Y);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se cambia el grado del polinomio %%%

    
    for fold=1:Rept
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(N,'Kfold',Rept);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold),:);
        Ytest=Y(particion.test(fold));
        

        m = length(Ytrain); % Number of training examples
        n=length(Ytest);
         %%% Normalización %%%
    
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
         %%%  %%%
        Modelo = ClassificationDiscriminant.fit(Xtrain,Ytrain,'discrimType','diagLinear');
        Yest = Modelo.predict(Xtest);
        % Compute the confusion matrix
        C_D = confusionmat(Ytest,Yest);
        % Examine the confusion matrix for each class as a percentage of the true class
        C_D = bsxfun(@rdivide,C_D,sum(C_D,2)) * 100;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
    end
    Eficiencia = mean(EficienciaTest);
    Error=1-Eficiencia;
    IC = std(EficienciaTest);
    MatrizConfusion
    Texto=['La eficiencia de discriminantes Guassina obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    Texto=['El error de clasificación en prueba es: ',num2str(Error)];
    disp(Texto);
    
elseif punto==2
   %%%k vecinos Cercanos%%%
    fprintf('k vecinos Cercanos')
   %%% Se crean los datos de forma aleatoria %%%
    load('DataTest.mat');
    N=size(X,1);
    Rept=1;
    N=N*0.1;
  
    Y(Y<=1000)=1;
    Y(Y>1000 & Y<=2000)=2;
    Y(Y>2000)=3;
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
    EficienciaTest=zeros(1,Rept);

    unqY=unique(Y);
    countElY=histc(Y,unqY);
    relFreqY=countElY/numel(Y);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se cambia el grado del polinomio %%%

    
    for fold=1:Rept
    
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(N,'Kfold',Rept);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold),:);
        Ytest=Y(particion.test(fold));
        

        m = length(Ytrain); % Number of training examples
        n=length(Ytest);
         %%% Normalización %%%
    
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
         %%%  %%%
        k=10;
        Yest=vecinosCercanos(Xtest,Xtrain,Ytrain,k,'class'); 
        % Compute the confusion matrix
        C_knn = confusionmat(Ytest,Yest);
        % Examine the confusion matrix for each class as a percentage of the true class
        C_knn = bsxfun(@rdivide,C_knn,sum(C_knn,2)) * 100;
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
    end
   
    Eficiencia = mean(EficienciaTest);
    Error=1-Eficiencia;
    IC = std(EficienciaTest);
    MatrizConfusion
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    Texto=['El error de clasificación en prueba es: ',num2str(Error)];
    disp(Texto);
elseif punto==3
    %%%Redes Neuronales%%%
    fprintf('Redes Neuronales')
    %%% Se crean los datos de forma aleatoria %%%
    load('DataTest.mat');
    N=size(X,1);
    Rept=10;
    N=N*0.3;
  
    Y(Y<=1000)=1;
    Y(Y>1000 & Y<=2000)=2;
    Y(Y>2000)=3;
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
    EficienciaTest=zeros(1,Rept);

    unqY=unique(Y);
    countElY=histc(Y,unqY);
    relFreqY=countElY/numel(Y);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for fold=1:Rept
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(N,'Kfold',Rept);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold),:);
        Ytest=Y(particion.test(fold));
 
         %%% Normalización %%%
    
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
         %%%  %%%
        [~, net] = neuronaln(Xtrain,Ytrain);
         %%% Validación de los modelos. %%%
         % Make a prediction for the test set
          Yest = net(Xtest');
          Yest = round(Yest');
      
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
    end
   
    Eficiencia = mean(EficienciaTest);
    Error=1-Eficiencia;
    IC = std(EficienciaTest);
    MatrizConfusion
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    Texto=['El error de clasificación en prueba es: ',num2str(Error)];
    disp(Texto);
elseif punto==4    
    %%% Random Forest %%%
    fprintf('Random Forest...\n');
    load('DataTest.mat');
    N=size(X,1);
    Rept=10;
    N=N*0.3;
    Y(Y<=1000)=1;
    Y(Y>1000 & Y<=2000)=2;
    Y(Y>2000)=3;
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
    EficienciaTest=zeros(1,Rept);

    unqY=unique(Y);
    countElY=histc(Y,unqY);
    relFreqY=countElY/numel(Y);
    for fold=1:Rept
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(N,'Kfold',Rept);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold),:);
        Ytest=Y(particion.test(fold));
 
         %%% Normalización %%%
    
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
        
        
        NumArboles=5;
        Modelo=entrenarFOREST(NumArboles,Xtrain,Ytrain);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %%% Validación de los modelos. %%%
        Yest=testFOREST(Modelo,Xtest);
        Yest = round(Yest);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
    end
   
    Eficiencia = mean(EficienciaTest);
    Error=1-Eficiencia;
    IC = std(EficienciaTest);
    MatrizConfusion
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    Texto=['El error de clasificación en prueba es: ',num2str(Error)];
    disp(Texto);
    
elseif punto==5
    %%% Se crean los datos de forma aleatoria %%%
    boxConstraint=100;
    gamma=100;
    load('DataTest.mat');
    N=size(X,1);
    Rept=10;
    N=N*0.3;
  
    Y(Y<=1000)=1;
    Y(Y>1000 & Y<=2000)=2;
    Y(Y>2000)=3;
    NumClases=length(unique(Y)); %%% Se determina el número de clases del problema.
    EficienciaTest=zeros(1,Rept);

    unqY=unique(Y);
    countElY=histc(Y,unqY);
    relFreqY=countElY/numel(Y);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se cambia el grado del polinomio %%%

    
    for fold=1:Rept
        %%% Se hace la partición de las muestras %%%
        %%%      de entrenamiento y prueba       %%%
        rng('default');
        particion=cvpartition(N,'Kfold',Rept);
        Xtrain=X(particion.training(fold),:);
        Xtest=X(particion.test(fold),:);
        Ytrain=Y(particion.training(fold),:);
        Ytest=Y(particion.test(fold));
 
         %%% Normalización %%%
    
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
      
        
        Modelo = cell(NumClases,1);
        targets = zeros(size(Ytrain,1),NumClases);
        
        for i=1:NumClases % se crea un modelo por cada clase
            Ytraini = Ytrain;
            Ytraini(Ytraini~=i)=-1;
            Ytraini(Ytraini==i)=1;
            targets(:,i) = Ytraini; % los targets son 1 para la clase y -1 en otro caso
            Modelo{i} = entrenarSVM(Xtrain,targets(:,i),'c',boxConstraint,gamma,'RBF_kernel');
        end
        numTest = size(Ytest,1);
        % Se realizan las predicciones usando los 3 clasificadores
        Yestall = zeros(numTest,NumClases);
        for j=1:NumClases
            Yestall(:,j) = testSVM(Modelo{j},Xtest); % se obtiene la predicción de cada modelo
        end
         % Se crea el vector de predicciones con las clases a la que
         % pertenece cada muestra. Luego se procede a actualizar aquellas
         % que presentan conflicto
        [~,Yest] = max(Yestall,[],2);
        %Se evaluan todas las muestras de prueba para determinar conflictos
        for i=1:numTest
            % se encuentran las predicciones iguales para una misma 
            % muestra por medio de un histograma
            [n, bin] = histc(Yestall(i,:), unique(Yestall(i,:))); 
            multiple = find(n > 1);
            index = find(ismember(bin, multiple)); % clases en conflicto
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            numEquals = size(index,2); % numero de clases en conflicto
            
            % Si el valor en conflicto de la prediccion es 1, o si hay 3
            % valores iguales (en el caso de los tres en -1), se procede
            % a resolver el conflicto con la evaluacion de la funcion
            if Yestall(i,index(1,1)) == 1 || numEquals == NumClases 
                evalY = zeros(1,numEquals);
                for l=1:numEquals
                    indices=find(Modelo{index(l)}.selector); % indices de los vectores de soporte
                    SupportVectors = Xtrain(indices,:); % vectores de soporte
                    SupportVectorsTargets=targets(indices,index(l)); % targets de los vectores de soporte
                    evalY(1,l)= evaluarFuncioSVM(Modelo{index(l)}.alpha,...
                        Modelo{index(l)}.b,SupportVectorsTargets,...
                        SupportVectors,Xtest(i,:),gamma,'gauss'); % se evalua cada clase en conflicto
                end
                if numEquals == NumClases % si la muestra no quedo en ninguna clase (las 3 en -1)
                    [~,classInd] = min(evalY); % se escoge la mas cercana
                else % si la muestra quedo en más de una clase
                    [~,classInd] = max(evalY); % se escoge la más lejana
                end
                Yest(i) = classInd; % se actualiza el vector con las decisiones
            end
        end

        % Compute the confusion matrix
        C_nn = confusionmat(Ytest,Yest);
        % Examine the confusion matrix for each class as a percentage of the true class
        C_nn = bsxfun(@rdivide,C_nn,sum(C_nn,2)) * 100
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        MatrizConfusion=zeros(NumClases,NumClases);
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i))=MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        EficienciaTest(fold)=sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
            
    end
   
    Eficiencia = mean(EficienciaTest);
    Error=1-Eficiencia;
    IC = std(EficienciaTest);
    MatrizConfusion
    Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia),' +- ',num2str(IC)];
    disp(Texto);
    Texto=['El error de clasificación en prueba es: ',num2str(Error)];
    disp(Texto);
    
end

