%%% the dummy regressor predicts the value by mean value of y

classdef LinearRegressorLocalWeight < model.regressor.LinearRegressor
    properties
        X; % the X value we predict  
        y; % the y value we predict
    end
    
    methods
        function linearRegressorLocalWeightObj = LinearRegressorLocalWeight (X,y)  
           linearRegressorLocalWeightObj=linearRegressorLocalWeightObj@model.regressor.LinearRegressor(X);
           linearRegressorLocalWeightObj.X = X;
           linearRegressorLocalWeightObj.y = y;
        end
 
        function predictedValue = predict (obj,X,cfg)
            A = [ones(size(X,1),1) X];
            B = [ones(size(obj.X,1),1) obj.X];            
            L = zeros(length(obj.X));
            for i = 1:length(X)
                for j = 1:length(obj.X)
                    L(j,j)=exp(-((X(i)-obj.X(j))^2)/(2*cfg('tau')^2))/2;
                end
                w = (transpose(B)*L*B)\transpose(B)*L*obj.y;
                predictedValue(i) = A(i,:) * w;
            end    

        end
    end
    methods (Static)
        function linearRegressorLocalWeightObj = train (X, y)
            linearRegressorLocalWeightObj = model.regressor.LinearRegressorLocalWeight(X,y);
        end
    end
end

