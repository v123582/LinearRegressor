%%% the dummy regressor predicts the value by mean value of y

classdef LinearRegressor < handle
   properties
      w; % the w value we predict
   end
   
   methods
       function linearRegressorObj = LinearRegressor (w)  % constructor
           linearRegressorObj.w = w;
       end
       function predictedValue = predict(obj, X)
           X = [ones(size(X,1),1) X];
            predictedValue = X * obj.w;
       end
   end
   
   methods (Static)
      function LinearRegressorObj = train (X, y)
        X = [ones(size(X,1),1) X];
        LinearRegressorObj = model.regressor.LinearRegressor(inv(transpose(X)*X)*transpose(X)*y);

      end
   end
end