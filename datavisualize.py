from bokeh.plotting import figure, show
from bokeh.sampledata.iris import flowers
import pandas as pd
import numpy as np

class DataSet():
    '''Document'''
    def __init__(self, csvfile):
        self.data = pd.read_csv(csvfile)
        self.xs=self.data.columns[0]
        self.ys=self.data.columns[1:]
        self.xdata=self.data[self.data.columns[0:1]]
        self.ydata=self.data[self.data.columns[1:]]

    def xnames(self):
        #print(self.xs)
        return(self.xs)

    def ynames(self):
        #print(self.ys)
        return(self.ys)

    def xvalues(self, names):
        #print(self.xdata[names])
        return(self.xdata[names])

    def yvalues(self,names):
        #print(self.ydata[names])
        return(self.ydata[names])


ideal = DataSet("ideal.csv")
train = DataSet("train.csv")
test = DataSet("test.csv")

colors = np.array([ [x, x, x] for x in range(50,100) ], dtype="uint8")
print(colors)
idealdata=pd.DataFrame()
for y in ideal.ynames():
    data = pd.DataFrame()
    data['x'] = ideal.xvalues(ideal.xnames())
    data['y'] = ideal.yvalues([y])
    data = data.assign(ys=[y for x in range(400)])
    idealdata = idealdata.append(data, ignore_index=True)

traindata=pd.DataFrame()
for y in train.ynames():
    data = pd.DataFrame()
    data['x'] = train.xvalues(train.xnames())
    data['y'] = train.yvalues([y])
    data = data.assign(ys=[y for x in range(400)])
    traindata = traindata.append(data, ignore_index=True)

testdata=pd.DataFrame()
for y in test.ynames():
    data = pd.DataFrame()
    data['x'] = test.xvalues(test.xnames())
    data['y'] = test.yvalues([y])
    data = data.assign(ys=[y for x in range(100)])
    testdata = testdata.append(data, ignore_index=True)

#print(idealdata)
#print(traindata)
#print(testdata)


from bokeh.transform import factor_cmap
from bokeh.palettes import all_palettes,  mpl,d3, Magma256, Magma, Inferno, Plasma, Viridis256, Cividis256
import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.models import DataRange1d
from bokeh.layouts import gridplot


#palette = Magma256[len(idealdata['ys'].unique())]
#palette = all_palettes['Magma256'][len(idealdata['ys'].unique())]
color_map = bmo.CategoricalColorMapper(factors=idealdata['ys'].unique(),
                                   palette=Magma256)
ideal_cmap = factor_cmap('ys', palette=Cividis256, factors=sorted(idealdata.ys.unique()))
train_cmap = factor_cmap('ys', palette=Inferno[4], factors=sorted(traindata.ys.unique()))
test_cmap = factor_cmap('ys', palette=Magma[4], factors=sorted(testdata.ys.unique()))

idelfig = figure(plot_width=600, plot_height=500, title = "Ideal Dataset Visualization")
idelfig.scatter('x','y',source=idealdata,fill_alpha=1.0, fill_color=ideal_cmap,size=10,legend='ys')
idelfig.xaxis.axis_label = 'Credit Rating'
idelfig.yaxis.axis_label = 'Possible Default Amount'
#idelfig.legend.location = "toidelfig_left"
idelfig.legend.orientation = "horizontal"

trainfig = figure(plot_width=600, plot_height=500, title = "Training Dataset Visualization")
trainfig.scatter('x','y',source=traindata,fill_alpha=1.0, fill_color=train_cmap,size=10,legend='ys')
trainfig.xaxis.axis_label = 'Credit Rating'
trainfig.yaxis.axis_label = 'Possible Default Amount'
#trainfig.legend.location = "totrainfig_left"
trainfig.legend.orientation = "horizontal"

testfig = figure(plot_width=600, plot_height=500, title = "Test Dataset Visualization")
testfig.scatter('x','y',source=testdata,fill_alpha=1.0, fill_color=test_cmap,size=10,legend='ys')
testfig.xaxis.axis_label = 'Credit Rating'
testfig.yaxis.axis_label = 'Possible Default Amount'
#testfig.legend.location = "totestfig_left"
testfig.legend.orientation = "horizontal"
p=gridplot([[idelfig, trainfig, testfig],[None,None,None]])
show(p)
#exit()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ideal_models = {}
X_ideal = ideal.xvalues(ideal.xnames()).to_numpy()
X_ideal = np.reshape(X_ideal, (X_ideal.shape[0],1))
#print("X",X_ideal.shape)
for y in ideal.ynames():
    y_ideal = ideal.yvalues([y]).values
    #print(y, y_ideal.shape)
    model =  LinearRegression()
    ideal_models[y] =  model.fit(X_ideal, y_ideal)
    print(ideal_models[y].score(X_ideal, y_ideal))

dftrain = pd.DataFrame()
ideal_train_map = {}

X_train_plt = train.xvalues(train.xnames())
#X_train_plt = train.xvalues(train.xnames()).to_numpy()
X_train = X_train_plt.to_numpy()
X_train = np.reshape(X_train, (X_train.shape[0],1))
for yname_train in train.ynames():
    y_train = train.yvalues([yname_train]).values
    rmses=[]
    for y in ideal.ynames():
        y_pred = ideal_models[y].predict(X_train)
        rmse = mean_squared_error(y_train, y_pred)
        #rmse = ideal_models[y].score(X_train, y_train)
        rmses.append(rmse)
    #print(rmses)
    dftrain[yname_train] = rmses
    ideal_train_map[yname_train] = dftrain[[yname_train]].idxmin()

#print(dftrain)
ideal_train_map = dftrain.idxmin().to_dict()
for key in ideal_train_map.keys():
    ideal_train_map[key] = "y"+str(ideal_train_map[key]+1)
print(ideal_train_map)

trainfigs=[]
for yname_train in train.ynames():
    y_train = train.yvalues([yname_train]).values
    y_pred = ideal_models[ideal_train_map[yname_train]].predict(X_train)
    print(y_train.flatten().shape, y_pred.flatten().shape)
    print(mean_squared_error(y_train, y_pred))
    trainfig= figure(plot_width=450, plot_height=400, title = "Actual v/s Predicted Default Amount for "+yname_train)
    trainfig.y_range = DataRange1d(end=150)
    trainfig.xaxis.axis_label = 'Credit Rating'
    trainfig.yaxis.axis_label = 'Possible Default Amount'
    trainfig.scatter(X_train_plt, y_train.flatten(), size=5, legend_label="Actual Default Amount")
    trainfig.line(X_train_plt, y_pred.flatten(), line_width=3,line_color='red', legend_label="Predicted Default Amount")
    trainfigs.append(trainfig)

q=gridplot([[trainfigs[0], trainfigs[1]],[trainfigs[2], trainfigs[3]]])
show(q)

dftrain.to_csv("ideal_train_mse.csv", index=False)

X_test_plt = test.xvalues(test.xnames())
#X_test_plt = test.xvalues(test.xnames()).to_numpy()
X_test = X_test_plt.to_numpy()
X_test = np.reshape(X_test, (X_test.shape[0],1))
y_test = test.yvalues(test.ynames()).values
testfigs=[]
for yname_train in train.ynames():
    y_pred = ideal_models[ideal_train_map[yname_train]].predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print(yname_train, rmse)
    print(y_test.flatten().shape, y_pred.flatten().shape)
    testfig= figure(plot_width=500, plot_height=400, title = "Actual v/s Predicted Default Amount for "+ideal_train_map[yname_train]+" ideal function")
    testfig.y_range = DataRange1d(end=150)
    testfig.xaxis.axis_label = 'Credit Rating'
    testfig.yaxis.axis_label = 'Possible Default Amount'
    testfig.scatter(X_test_plt, y_test.flatten(), size=5, legend_label="Actual Default Amount")
    testfig.line(X_test_plt, y_pred.flatten(), line_width=3,line_color='red', legend_label="Predicted Default Amount")
    testfigs.append(testfig)

r=gridplot([[testfigs[0], testfigs[1]],[testfigs[2], testfigs[3]]])
show(r)


exit()

