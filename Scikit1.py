# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:51:15 2018

@author: ccl100047
"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
import plotly.offline as py
import plotly.graph_objs as go

datatable= pd.read_csv('Testdata.csv')

X_train= datatable[['EWT','OAT','Comp_%']]
y_train= datatable['EXV']

mlp=MLPRegressor(hidden_layer_sizes=(30,30,30), activation='relu', solver='lbfgs', 
                  alpha=0.0001, batch_size='auto', learning_rate='constant', 
                  learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, 
                  random_state=1, tol=0.0001, verbose=False, warm_start=False, 
                  momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                  validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
mlp.fit(X_train, y_train)                         
print (mlp.n_layers_)
print (mlp.n_iter_)
print (mlp.loss_)
print (mlp.out_activation_)
print (mlp.score(X_train,y_train))

Pre=mlp.predict(X_train)

trace1 = go.Scatter(
     y = y_train.values.ravel(),
  name = 'Test'
)       
trace2 = go.Scatter(
     y = Pre,
  name = 'Predict'
)
data = [trace1,trace2]
layout = dict(
      title = 'EXV',
      yaxis = dict(title = 'EXV (%)', range=[0,100])
)

fig = dict(data=data, layout=layout)
#py.plot(fig, filename='ExvComp.html')

#Using Office1 FS to test
datatable1= pd.read_csv('OFF_FS1.csv')
X_test1= datatable1[['EWT','OAT','Comp_%']]
y_test1= datatable1['EXV']

Pre1=mlp.predict(X_test1)
trace3 = go.Scatter(
     y = y_test1.values.ravel(),
  name = 'Real'
)       
trace4 = go.Scatter(
     y = Pre1,
  name = 'Predict'
)
data1 = [trace3,trace4]
layout1 = dict(
      title = 'Office FS01 EXV',
      yaxis = dict(title = 'EXV (%)', range=[0,100])
)
fig1 = dict(data=data1, layout=layout1)
py.plot(fig1, filename='ExvComp1.html')

#Using Office2 FS to test
datatable2= pd.read_csv('OFF_FS2.csv')
X_test2= datatable2[['EWT','OAT','Comp_%']]
y_test2= datatable2['EXV']

Pre2=mlp.predict(X_test2)
trace5 = go.Scatter(y = y_test2.values.ravel(),name = 'Real')       
trace6 = go.Scatter(y = Pre2,name = 'Predict')
data2 = [trace5,trace6]
layout2 = dict(
      title = 'Office FS02 EXV',
      yaxis = dict(title = 'EXV (%)', range=[0,100])
)
fig2 = dict(data=data2, layout=layout2)
py.plot(fig2, filename='ExvComp2.html')

#Using Office3 FS to test
datatable3= pd.read_csv('OFF_FS3.csv')
X_test3= datatable3[['EWT','OAT','Comp_%']]
y_test3= datatable3['EXV']

Pre3=mlp.predict(X_test3)
trace7 = go.Scatter(y = y_test3.values.ravel(),name = 'Real')       
trace8 = go.Scatter(y = Pre3,name = 'Predict')
data3 = [trace7,trace8]
layout3 = dict(
      title = 'Office FS03 EXV',
      yaxis = dict(title = 'EXV (%)', range=[0,100])
)
fig3 = dict(data=data3, layout=layout3)
py.plot(fig3, filename='ExvComp3.html')