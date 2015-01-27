import model.regressor.LinearRegressor
import model.regressor.LinearRegressorLocalWeight

clear %clear workspace

%--- TODO: please import the dataset here ---%
 X=load('data/X.dat');
 y=load('data/Y.dat');
%--- TODO: modify the DummyRegressor to your LinearRegressor & LinearRegressorLocalWeight ---%
%---       please follow the specs strickly                              ---%
myKeys = {'tau'};
myValues = [100];
cfg = containers.Map(myKeys,myValues);

myRegressor1 = LinearRegressor.train(X,y);
value1 = myRegressor1.predict(X);
myRegressor2 = LinearRegressorLocalWeight.train(X,y);
value2 = myRegressor2.predict(X,cfg);
value2 = transpose(value2)
%%% plot data %%%
scatter (X,y,'g');
hold on;

%--- TODO: plot the regressor you train ---%
scatter (X,value2,'r');
plot(X, value1, 'r');
hold off;

