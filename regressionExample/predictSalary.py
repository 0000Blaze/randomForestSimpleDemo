import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
dataset = pd.read_csv('./regressionExample/Position_Salaries.csv')
dataset.head()
 
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
 
# for 10 trees
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X,y)
 
y_pred=regressor.predict([[6.5]])
y_pred
print("10 trees prediction:",y_pred) 

#higher resolution graph
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1) 
  
plt.scatter(X,y, color='red') #plotting real points
plt.plot(X_grid, regressor.predict(X_grid),color='blue') #plotting for predict points
  
plt.title("Truth or Bluff(Random Forest - Smooth)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 
 
# for 100 trees
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X,y)
 
#higher resolution graph
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,y, color='red') 
  
plt.plot(X_grid, regressor.predict(X_grid),color='blue') 
plt.title("Truth or Bluff(Random Forest - Smooth)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 
y_pred=regressor.predict([[6.5]])
y_pred
print("100 trees prediction:",y_pred)

# for 300 trees
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)
 
#higher resolution graph
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1) 
  
plt.scatter(X,y, color='red') #plotting real points
plt.plot(X_grid, regressor.predict(X_grid),color='blue') #plotting for predict points
  
plt.title("Truth or Bluff(Random Forest - Smooth)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 
y_pred=regressor.predict([[6.5]])
y_pred
print("300 trees prediction:",y_pred)