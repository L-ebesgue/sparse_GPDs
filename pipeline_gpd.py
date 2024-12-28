import sys
import numpy as np
import numpy.random as npr
import pandas as pd
import pickle as pck
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
from os.path import expanduser
from time import time

import multipers as mp
import multipers.data.graphs as mdg
import multipers.ml.signed_measures as mms
import multipers.ml.point_clouds as mmp
from multipers.distances import sm_distance
from multipers.ml.signed_measures import SignedMeasure2Convolution
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from sklearn.preprocessing   import LabelEncoder, MinMaxScaler
from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.pipeline        import Pipeline
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import scipy.ndimage as sci
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.functional as nnf

from utils import GPD

dataset_path = sys.argv[1]
dataset = dataset_path.split('/')[-1]
n_full = int(sys.argv[2])
N_sparse, N_full = int(sys.argv[3]), int(sys.argv[4])
n_epochs = int(sys.argv[5])
lr = float(sys.argv[6])
gpd_resolution = int(sys.argv[7])
gpd_window = int(sys.argv[8])
gpd_bandwidth = float(sys.argv[9])
homology_dimensions = [int(h) for h in sys.argv[10].split('-')]

normalize = True
gpd_dimension = 6
prm_str = '_'.join([sys.argv[i] for i in [3,4,5,6,10]])

class GPD2Convolution(BaseEstimator, TransformerMixin):

    # only gaussian kernels for now
    def __init__(self, window, bandwidth, resolution, dimension=gpd_dimension):
        self.window, self.bandwidth, self.resolution, self.dimension = window, bandwidth, resolution, dimension
        self.bins = [self.resolution for _ in range(self.dimension)]

    def fit(self, X, y=None):
        Xstack = np.vstack(X)
        if len(Xstack) > 0:
            self.m, self.M = Xstack.min(axis=0), Xstack.max(axis=0)
        else:
            self.m, self.M = np.zeros([self.dimension]), np.zeros([self.dimension])
        self.range = [(self.m[d], self.M[d]) for d in range(self.dimension)]
        return self

    def transform(self, X):
        Xfit = []
        for gpd in X:
            gpd_use = np.empty([0,self.dimension]) if len(gpd) == 0 else gpd             
            histo, _ = np.histogramdd(gpd_use, bins=self.bins, range=self.range)
            convolution = sci.gaussian_filter(histo, sigma=self.bandwidth, radius=self.window)
            Xfit.append(convolution.flatten()[None,:])
        return np.vstack(Xfit)

def cartesprod(a,b):
    a_ = torch.reshape(torch.tile(a, [1, b.shape[0]]), (a.shape[0] * b.shape[0], a.shape[1]))
    b_ = torch.tile(b, [a.shape[0], 1])
    return torch.cat([a_, b_], 1)

def max_pair(X1, X2, Y1, Y2):
    XD = torch.mul(  torch.where(X1 <= X2, 1, 0), torch.abs(X2-X1)  )
    YD = torch.mul(  torch.where(Y1 <= Y2, 1, 0), torch.abs(Y2-Y1)  )
    MXY = torch.max(torch.cat([XD,YD], dim=1), dim=1, keepdim=True)[0]
    return MXY

def min_max_pair(X1,X2,Y1,Y2,  X3,X4,Y3,Y4):
    MXY12 = max_pair(X1, X2, Y1, Y2)
    MXY34 = max_pair(X3, X4, Y3, Y4)
    mXY = torch.min(torch.cat([MXY12, MXY34], dim=1), dim=1, keepdim=True)[0]
    return mXY

def min_uv_max_pair(X1,X2,Y1,Y2,  X3,X4,Y3,Y4, U,V):
    MXY12 = max_pair(X1, X2, Y1, Y2)
    MXY34 = max_pair(X3, X4, Y3, Y4)
    mXY = torch.min(torch.cat([MXY12, MXY34, U, V], dim=1), dim=1, keepdim=True)[0]
    return mXY

def max_min_pair(U,V,W,X):
    mUV = torch.min(torch.cat([U,V], dim=1), dim=1, keepdim=True)[0]
    mWX = torch.min(torch.cat([W,X], dim=1), dim=1, keepdim=True)[0]
    MUVWX = torch.max(torch.cat([mUV, mWX], dim=1), dim=1, keepdim=True)[0]
    return MUVWX

def upper_bound(sparse_grid, full_grid, fast=True):

    # Fast but requires a lot of RAM
    if fast:

        Ns, Nf = sparse_grid.shape[0], full_grid.shape[0]
#        print(Ns, Nf)

        Is = cartesprod(sparse_grid, full_grid)
        Is1, Is2 = Is[:,:6], Is[:,6:]
        x1, x2, x1mb, x2mf, x1pd, x2ph = Is1[:,0:1], Is2[:,0:1], Is1[:,0:1]-Is1[:,3:4], Is2[:,0:1]-Is2[:,3:4], Is1[:,0:1]+Is1[:,5:6], Is2[:,0:1]+Is2[:,5:6]
        y1, y2, y1mc, y2mg, y1pa, y2pe = Is1[:,1:2], Is2[:,1:2], Is1[:,1:2]-Is1[:,4:5], Is2[:,1:2]-Is2[:,4:5], Is1[:,1:2]+Is1[:,2:3], Is2[:,1:2]+Is2[:,2:3]
        # h_a, h_bd, h_d, h_ac = Is1[:,2:3]/2, (Is1[:,3:4]+Is1[:,5:6])/2, Is1[:,5:6]/2, (Is1[:,2:3]+Is1[:,4:5])/2
        # h_e, h_fh, h_h, h_eg = Is2[:,2:3]/2, (Is2[:,3:4]+Is2[:,5:6])/2, Is2[:,5:6]/2, (Is2[:,2:3]+Is2[:,4:5])/2
   
        yellow = torch.max(torch.cat([min_max_pair(x2mf,x1mb,y2,y1,  x2mf,x1,y2,y1mc), 
                                      min_max_pair(x2,x1mb,y2mg,y1,  x2,x1,y2mg,y1mc)], dim=1), dim=1, keepdim=True)[0]

        green = max_pair(x1pd,x2ph,y1pa,y2pe)

        # cyan = torch.max(torch.cat([min_uv_max_pair(x1mb,x2mf,y1,y2,  x1mb,x2,y1,y2mg,  h_a,h_bd), 
        #                             min_uv_max_pair(x1,x2mf,y1mc,y2,  x1,x2,y1mc,y2mg,  h_d,h_ac)], dim=1), dim=1, keepdim=True)[0]

        # red = torch.min(torch.cat([max_pair(x2ph,x1pd,y2pe,y1pa),
        #                            max_min_pair(2*h_bd,2*h_a,2*h_d,2*h_ac)/2], dim=1), dim=1, keepdim=True)[0]

        grey = torch.max(torch.cat([min_max_pair(x1mb,x2mf,y1,y2,  x1mb,x2,y1,y2mg), 
                                    min_max_pair(x1,x2mf,y1mc,y2,  x1,x2,y1mc,y2mg)], dim=1), dim=1, keepdim=True)[0]

        purple = max_pair(x2ph,x1pd,y2pe,y1pa)

        # blue = torch.max(torch.cat([min_uv_max_pair(x2mf,x1mb,y2,y1,  x2mf,x1,y2,y1mc,  h_fh,h_e), 
        #                             min_uv_max_pair(x2,x1mb,y2mg,y1,  x2,x1,y2mg,y1mc,  h_h,h_eg)], dim=1), dim=1, keepdim=True)[0]

        # brown = torch.min(torch.cat([max_pair(x1pd,x2ph,y1pa,y2pe),
        #                              max_min_pair(2*h_fh,2*h_e,2*h_h,2*h_eg)/2], dim=1), dim=1, keepdim=True)[0]

#        print([yellow, green, cyan, red, grey, purple, blue, brown])
        Emax = torch.max(torch.cat([yellow, green, grey, purple], dim=1), dim=1)[0]
        # Emax = torch.max(torch.cat([yellow, green, cyan, red, grey, purple, blue, brown], dim=1), dim=1)[0]
        E = torch.reshape(Emax, [Ns, Nf])
#        print(E)

    # Light on RAM but very long
    else:
        print('Not implemented yet')
        return 0

    Erows, Ecols = torch.min(E, dim=0)[0], torch.min(E, dim=1)[0]
    E1, E2 = torch.max(Erows, dim=0, keepdim=True)[0], torch.max(Ecols, dim=0, keepdim=True)[0]
    return torch.max(torch.cat([E1, E2]))

def compute_GPD(st, intervals, hd, norm):
    mfilt = list(st.get_simplices())
    if norm == True:
        filt0 = [filts[0] for simplex, filts in mfilt]
        filt0 = MinMaxScaler().fit_transform(np.array(filt0)[:,None]).flatten()
        filt1 = [filts[1] for simplex, filts in mfilt]
        filt1 = MinMaxScaler().fit_transform(np.array(filt1)[:,None]).flatten()
        mfilt = [(simplex, [filt0[i], filt1[i]]) for i, (simplex,_) in enumerate(mfilt)]
#    print(mfilt, intervals)
    measure = GPD(mfilt, intervals, hd)
    gpd_pos, gpd_neg = [], []
    for interval, multiplicity in measure.items():
        if multiplicity != 0:
            interval = np.array(interval)[None,:]
            if multiplicity > 0:
                [gpd_pos.append(interval) for _ in range(multiplicity)]
            else:
                [gpd_neg.append(interval) for _ in range(multiplicity)]
    gpd_pos = np.vstack(gpd_pos) if len(gpd_pos) > 0 else np.empty([0,gpd_dimension])
    gpd_neg = np.vstack(gpd_neg) if len(gpd_neg) > 0 else np.empty([0,gpd_dimension])
    return gpd_pos, gpd_neg

def generate_score(simplextrees_train, simplextrees_test, intervals, norm, model_name, output_file):
    n_train = len(simplextrees_train)

    print('Computing ' + model_name + ' gpds on data points...')
    start = time()
    gpd_train_pos, gpd_train_neg, gpd_test_pos, gpd_test_neg = [], [], [], []
    for idx, st in enumerate(simplextrees_train + simplextrees_test):
        print(idx)
        gpd_pos_hlist, gpd_neg_hlist = [], []
        for h in homology_dimensions:
            gpd_pos, gpd_neg = compute_GPD(st[0], intervals, h, norm)
            gpd_pos_hlist.append(gpd_pos)
            gpd_neg_hlist.append(gpd_neg)
        #print(gpd_pos_hlist, gpd_neg_hlist)
        gpd_pos, gpd_neg = np.vstack(gpd_pos_hlist), np.vstack(gpd_neg_hlist)
        if idx < n_train:
            gpd_train_pos.append(gpd_pos)
            gpd_train_neg.append(gpd_neg)
        else:
            gpd_test_pos.append(gpd_pos)
            gpd_test_neg.append(gpd_neg)
        #print(gpd)
    end = time()
    print('Done')

    print('Learning ' + model_name + 'model...')

#    GPD2C = GPD2Convolution(window=gpd_window, bandwidth=gpd_bandwidth, resolution=gpd_resolution, dimension=gpd_dimension)
#    conv_train = np.reshape(GPD2C.fit_transform(gpd_train_pos), [n_train, -1]) - np.reshape(GPD2C.fit_transform(gpd_train_neg), [n_train, -1])
#    conv_test = np.reshape(GPD2C.transform(gpd_test_pos), [n_test, -1]) - np.reshape(GPD2C.transform(gpd_test_neg), [n_test, -1])
#    classifier = RandomForestClassifier()
#    classifier = classifier.fit(conv_train, ytrain)

    pipe = Pipeline([("gpd2c",     GPD2Convolution(window=gpd_window, bandwidth=gpd_bandwidth, resolution=gpd_resolution, dimension=gpd_dimension)),
                     ("estimator", RandomForestClassifier(random_state=123))])
    param =    [
                {"gpd2c__window":         [1,2,3,4,5],
                 "gpd2c__bandwidth":      [10**p for p in range(-3,4)],
                 "gpd2c__resolution":     [5,10],
                 "gpd2c__dimension":      [6],
                 "estimator":             [RandomForestClassifier(random_state=123)]}
               ]
    skf = StratifiedKFold(n_splits=3, random_state=123, shuffle=True)
    splits = skf.split(np.ones([n_train,1]), ytrain)
    splits_l = list(splits)
#    for i, (train_index, test_index) in enumerate(splits_l):
#        print(f"Fold {i}:")
#        print(f"  Train: index={train_index}")
#        print(f"  Test:  index={test_index}")
    classifier = GridSearchCV(pipe, param, cv=splits_l)
    classifier = classifier.fit(gpd_train_pos, ytrain)
#    print(classifier.cv_results_, classifier.best_index_)
    print('Done')

    score = classifier.score(gpd_test_pos, ytest)

    print('RF on ' + model_name + ' gpd convolutions with best params ' + str(classifier.best_params_))
    print(score)
    output_file.write('RF on ' + model_name + ' gpd convolutions with best params ' + str(classifier.best_params_))
    output_file.write('\n')
    output_file.write("Test score = " + str(score) + ', time = ' + str(end-start) + 's')
    output_file.write('\n')

    return score


#gpd2c = GPD2Convolution(window=5, bandwidth=1., resolution=10, dimension=6)
#pds = [np.random.randint(0, 4, size=[np.random.randint(1,10,1)[0], 6]) for _ in range(10)]
#convs = gpd2c.fit_transform(pds)
#print(convs.min(), convs.max())

##############################################################################################################################################################################################################
print('Reading data...')
xtrain = np.array(pd.read_csv(dataset_path+"_TRAIN.txt", delimiter='  ', header=None, index_col=None, engine='python'))
ytrain = LabelEncoder().fit_transform(xtrain[:,0])
xtrain = xtrain[:,1:]
xtest = np.array(pd.read_csv(dataset_path + "_TEST.txt", delimiter='  ', header=None, index_col=None, engine='python'))
ytest = LabelEncoder().fit_transform(xtest[:,0])
xtest = xtest[:,1:]
TDE = TimeDelayEmbedding(dim=3, delay=1, skip=1)
xtrain = TDE.transform(xtrain)
xtest = TDE.transform(xtest)
n_train, n_test = len(xtrain), len(xtest)
print('Done')
print('Converting data into (multi-parameter) simplex trees...')
PC2ST = mmp.PointCloud2FilteredComplex(bandwidths=[-.1], num_collapses=-2, complex="rips", expand_dim=2, sparse=None, threshold=-np.inf) #sparse=0.5
simplextrees_train = PC2ST.fit_transform(xtrain)
simplextrees_test  = PC2ST.transform(xtest)
print('Done')
##############################################################################################################################################################################################################



##############################################################################################################################################################################################################
if normalize == False:
    full_indices = np.random.choice(a=n_train, size=n_full, replace=False)
    grids_full = []
    for st in [simplextrees_train[idx] for idx in full_indices]:
        grid = st[0].get_filtration_grid(resolution=[N_full, N_full], grid_strategy='regular')
        grids_full.append(grid)
##############################################################################################################################################################################################################



##############################################################################################################################################################################################################
print('Optimizing to find sparse grid...')
if normalize == False:
    min_grid_0 = np.array([grids_full[i][0].min() for i in range(n_full)]).min()
    min_grid_1 = np.array([grids_full[i][1].min() for i in range(n_full)]).min()
    max_grid_0 = np.array([grids_full[i][0].max() for i in range(n_full)]).max()
    max_grid_1 = np.array([grids_full[i][1].max() for i in range(n_full)]).max()
else:
    min_grid_0, max_grid_0, min_grid_1, max_grid_1 = 0., 1., 0., 1.

#print(min_grid_0, max_grid_0, min_grid_1, max_grid_1)

np.random.seed(123)

N_full_hardcoded = 2
N_sparse_hardcoded = 2
even = True

if even:
    full_grid = [torch.tensor( np.linspace(start=min_grid_0, stop=max_grid_0, num=N_full), dtype=torch.float32, requires_grad=False ), # x
                 torch.tensor( np.linspace(start=min_grid_1, stop=max_grid_1, num=N_full), dtype=torch.float32, requires_grad=False ), # y
                 torch.tensor( np.linspace(start=0., stop=(max_grid_1-min_grid_1)/4, num=N_full_hardcoded), dtype=torch.float32, requires_grad=False ), # a
                 torch.tensor( np.linspace(start=0., stop=(max_grid_0-min_grid_0)/4, num=N_full_hardcoded), dtype=torch.float32, requires_grad=False ), # b
                 torch.tensor( np.linspace(start=0., stop=(max_grid_1-min_grid_1)/4, num=N_full_hardcoded), dtype=torch.float32, requires_grad=False ), # c
                 torch.tensor( np.linspace(start=0., stop=(max_grid_0-min_grid_0)/4, num=N_full_hardcoded), dtype=torch.float32, requires_grad=False )  # d
                ]
    init_sparse_grid = [torch.tensor( np.linspace(start=min_grid_0, stop=max_grid_0, num=N_sparse), dtype=torch.float32, requires_grad=False ), # x
                        torch.tensor( np.linspace(start=min_grid_1, stop=max_grid_1, num=N_sparse), dtype=torch.float32, requires_grad=False ), # y
                        torch.tensor( np.linspace(start=0., stop=(max_grid_1-min_grid_1)/4, num=N_sparse_hardcoded), dtype=torch.float32, requires_grad=False ), # a
                        torch.tensor( np.linspace(start=0., stop=(max_grid_0-min_grid_0)/4, num=N_sparse_hardcoded), dtype=torch.float32, requires_grad=False ), # b
                        torch.tensor( np.linspace(start=0., stop=(max_grid_1-min_grid_1)/4, num=N_sparse_hardcoded), dtype=torch.float32, requires_grad=False ), # c
                        torch.tensor( np.linspace(start=0., stop=(max_grid_0-min_grid_0)/4, num=N_sparse_hardcoded), dtype=torch.float32, requires_grad=False )  # d
                       ]
else:
    full_grid = [torch.tensor( np.sort(npr.uniform(low=min_grid_0, high=max_grid_0, size=N_full)), dtype=torch.float32, requires_grad=False ), # x
                 torch.tensor( np.sort(npr.uniform(low=min_grid_1, high=max_grid_1, size=N_full)), dtype=torch.float32, requires_grad=False ), # y
                 torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_1-min_grid_1)/4, size=N_full_hardcoded)), dtype=torch.float32, requires_grad=False ), # a
                 torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_0-min_grid_0)/4, size=N_full_hardcoded)), dtype=torch.float32, requires_grad=False ), # b
                 torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_1-min_grid_1)/4, size=N_full_hardcoded)), dtype=torch.float32, requires_grad=False ), # c
                 torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_0-min_grid_0)/4, size=N_full_hardcoded)), dtype=torch.float32, requires_grad=False )  # d
                ]
    init_sparse_grid = [torch.tensor( np.sort(npr.uniform(low=min_grid_0, high=max_grid_0, size=N_sparse)), dtype=torch.float32, requires_grad=False ), # x
                        torch.tensor( np.sort(npr.uniform(low=min_grid_1, high=max_grid_1, size=N_sparse)), dtype=torch.float32, requires_grad=False ), # y
                        torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_1-min_grid_1)/4, size=N_sparse_hardcoded)), dtype=torch.float32, requires_grad=False ), # a
                        torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_0-min_grid_0)/4, size=N_sparse_hardcoded)), dtype=torch.float32, requires_grad=False ), # b
                        torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_1-min_grid_1)/4, size=N_sparse_hardcoded)), dtype=torch.float32, requires_grad=False ), # c
                        torch.tensor( np.sort(npr.uniform(low=0., high=(max_grid_0-min_grid_0)/4, size=N_sparse_hardcoded)), dtype=torch.float32, requires_grad=False )  # d
                       ]

init_sparse_grid = [sg.clone().detach() for sg in init_sparse_grid]
full_grid = torch.cartesian_prod(*full_grid)
init_sparse_grid = torch.cartesian_prod(*init_sparse_grid)
init_sparse_grid_numpy = init_sparse_grid.clone().detach().numpy()

#plt.figure()
#plt.scatter(init_sparse_grid_numpy[:,0], init_sparse_grid_numpy[:,1], s=10, label='init.')
#plt.grid()
#plt.legend()
#plt.title('min filt = ' + str(min_grid_0) + ', ' + str(min_grid_1) + ', max filt = ' + str(max_grid_0) + ', ' + str(max_grid_1) + ', size = ' + str(N_sparse) + ' x ' + str(N_sparse))
#plt.savefig('./results/' + dataset + '_init_grid_gpd_' + prm_str + '.png', format='png')

sparse_grid = init_sparse_grid.clone().detach().requires_grad_(True)
optimizer = optim.SGD([sparse_grid], lr=lr, momentum=0.9)
#optimizer = optim.Adam([sparse_grid], lr=lr) #, momentum=0.9)
scheduler = sched.ExponentialLR(optimizer, gamma=0.99)
sparse_grid_list, loss_list = [], []

for epoch in range(n_epochs):

    sparse_grid_clone = sparse_grid.clone()
    pair = [sparse_grid_clone.detach().numpy()]
    sparse_grid_clone[:,2:] = nnf.relu(sparse_grid_clone[:,2:]) # ensures the lengths are positive
    pair.append(sparse_grid_clone.detach().numpy())
    sparse_grid_list.append(pair)
    loss = upper_bound(sparse_grid_clone, full_grid, fast=True)
    loss_list.append(float(loss.detach().numpy()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

print('Done')

#print(loss_list)
plt.figure()
plt.plot(loss_list)
plt.grid()
plt.title('lr = ' + str(lr) + ', num epochs = ' + str(n_epochs) + ', n_full = ' + str(n_full))
plt.savefig('./results/' + dataset + '_optimization_loss_gpd_' + prm_str + '.png', format='png')

#print(init_sparse_grid)
#print(sparse_grid)
plt.figure()
plt.scatter(sparse_grid[:,0].detach().numpy(), sparse_grid[:,1].detach().numpy(), s=50, label='optim.')
plt.scatter(init_sparse_grid_numpy[:,0], init_sparse_grid_numpy[:,1], s=10, label='init.')
plt.grid()
plt.legend()
plt.title('min filt = ' + str(min_grid_0) + ', ' + str(min_grid_1) + ', max filt = ' + str(max_grid_0) + ', ' + str(max_grid_1) + ', size = ' + str(N_sparse) + ' x ' + str(N_sparse))
plt.savefig('./results/' + dataset + '_grids_gpd_' + prm_str + '.png', format='png')

pck.dump([loss_list, sparse_grid_list, init_sparse_grid, sparse_grid], open('./results/' + dataset + '_optim_gpd_' + prm_str + '.pkl', 'wb'))
##############################################################################################################################################################################################################



##############################################################################################################################################################################################################
output_file = open('./results/' + dataset + '_classification_results_gpd_' + prm_str + '.txt', 'w')    
init_score    = generate_score(simplextrees_train, simplextrees_test, init_sparse_grid, normalize, 'init_sparse',  output_file)
sparse_score  = generate_score(simplextrees_train, simplextrees_test, sparse_grid,      normalize, 'optim_sparse', output_file)
full_score    = generate_score(simplextrees_train, simplextrees_test, full_grid,        normalize, 'full',         output_file)
output_file.close()
##############################################################################################################################################################################################################

