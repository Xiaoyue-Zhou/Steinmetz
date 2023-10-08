## Package importation.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from scipy import stats

# for 3d plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# for animation
from matplotlib.animation import FuncAnimation

## Figure Setting
#  Figure settings
from matplotlib import rcParams

rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True

## function definition
def compute_acc(X, y, model):
    '''
    :param X: design matrix: trial * features
    :param y: label for each trial
    :param model: a glm model
    :return: acc (scalar)
    '''

    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    return acc

def plot_by_pretrial_feedback(s, feedback_type, title = None, normalize = False, save=False):
    '''
    :param s: spike data, (neuron * trial * time).
    :param feedback_type: an array, indicating the feedback information in the previous trial.
    :param title: title for the output fig.
    :param normalize: whether normalize the data by neuron in each trial. bool type.
    :param save: whether want to save the fig. bool type.
    :return: none
    '''

    isRew = feedback_type == 1
    isPen = feedback_type == -1

    lastIsRew = isRew[:-1]
    lastIsPen = isPen[:-1]

    if normalize:
        for i in range(s.shape[2]):
            s[:,:,i] = (s[:,:,i] - np.mean(s,2)) / np.std(s,2)

    s = s[:,1:,:]
    s_lastIsRew = s[:,lastIsRew,:]
    s_lastIsPen = s[:,lastIsPen,:]

    err_lastIsRew = np.std(np.mean(s_lastIsRew,0),0) / np.sqrt(s_lastIsRew.shape[1]-1)
    err_lastIsPen = np.std(np.mean(s_lastIsPen,0),0) / np.sqrt(s_lastIsPen.shape[1]-1)

    # draw the plot
    x = list(range(0,s.shape[2],1))
    plt.figure()
    myfig = plt.gcf()

    plt.errorbar(x, np.mean(s_lastIsRew, (0,1)), yerr=err_lastIsRew, label='Post Reward')
    plt.errorbar(x, np.mean(s_lastIsPen, (0,1)), yerr=err_lastIsPen, label='Post Penalty')

    plt.title(title)
    plt.legend()
    plt.show()

    if save:
      myfig.savefig('./'+ str(title))


def plot_by_curr_corr(s, corr, title = None, normalize = False, save=False):
    '''
    :param s: data of spike trians: (neurons * trials * time bins)
    :param corr: an array made up of 1s & -1s. 1 for correct and -1 for error
    :param title: the title of the plot generated. e.g., the name of brain region.
    :param normalize：whether normalize the spike data by neurons in each trial.
    :return: none (made by pyplot)
    '''

    isCorr = corr == 1
    isErr = corr == -1

    if normalize:
        for i in range(s.shape[2]):
            s[:,:,i] = (s[:,:,i] - np.mean(s,2)) / np.std(s,2)

    s = s[:,:,:]
    s_corr = s[:,isCorr,:]
    s_err = s[:,isErr,:]

    err_corr = np.std(np.mean(s_corr,0),0) / np.sqrt(s_corr.shape[1]-1)
    err_err = np.std(np.mean(s_err,0),0) / np.sqrt(s_err.shape[1]-1)

    # draw the plot
    x = list(range(0,s.shape[2],1))
    plt.figure()
    myfig = plt.gcf()

    plt.errorbar(x, np.mean(s_corr, (0,1)), yerr=err_corr, label='Correct final')
    plt.errorbar(x, np.mean(s_err, (0,1)), yerr=err_err, label='Error final')

    plt.title(title)
    plt.legend()
    plt.show()

    if save:
      myfig.savefig('./figure/'+ str(title))


# neural models
def neuronModel3D(s, y, dim_left = 'time',penalty='l2', C=1, cv=16):
    '''
    logistic regression, with cross validation and permutation test. permutate only once.

    :param s: spike data, (neuron * trial * time).
    :param y: label for each trial
    :param dim_left: indicate the colum in design matrix generated in the function. 'time' or 'neuron'.
    :param penalty:the penalty term used for logistic regression. 'none','l1',or'l2'.
    :param C: needed when there is penalty.
    :param cv: folds of cross validation
    :return: ori_acc,  scrambled_acc, cvdMat
    '''

    print('s shape =', s.shape)

    if dim_left == 'time':
      X = np.average(s, 0)
      print('X dim (time left):', X.shape)
    elif dim_left == 'neuron':
      X = np.average(s, 2)
      X = np.transpose(X)
      print('X dim (neuron left):', X.shape)
    else:
      print('Please enter "time" or "neuron" for the parameter: dim_left (set as "time" by default)')

    # add 1s line
    dim = X.shape[0]
    X = np.concatenate((np.ones(dim).reshape(dim,1),X), axis=1)
    print('add "1"s ->',X.shape)

    # fit GLM with regularization
    if penalty == 'l2':
        model = LogisticRegression(penalty='l2',C=C,max_iter=5000)
    elif penalty == 'l1':
        model = LogisticRegression(penalty='l1',C=C,max_iter=5000,solver='saga')
    else:
        model = LogisticRegression(penalty='none', max_iter=5000)

    model.fit(X,y)

    ori_acc = compute_acc(X, y, model) # a single number
    ori_cvd = cross_val_score(model, X, y, cv=cv) # an array (cv,)

    per = np.random.permutation(X.shape[0])
    X_scrambled = X[per,:]
    model.fit(X_scrambled, y)
    scrambled_acc = compute_acc(X_scrambled, y, model)
    scrambled_cvd = cross_val_score(model, X_scrambled, y, cv=cv)

    cvdMat = np.concatenate((ori_cvd.reshape(cv, 1), scrambled_cvd.reshape(cv, 1)), axis=1)

    return ori_acc,  scrambled_acc, cvdMat


def boot_strap(X, reference, each=100):  # generate new set of data
    '''
    do bootstrap to address frequency issue.
    :param X: original design matrix
    :param reference: original y. an array made up of 1s and -1s.
    :param each: how many trials will the bootstrap extract for each type of label.
    :return:
    '''

    idx1 = reference == 1
    idx2 = reference == -1

    X1 = X[idx1, :]
    row1 = X1[0]
    row1_choice = np.random.choice(row1, each, replace=True)
    X1_sample = X1[row1_choice, :]

    X2 = X[idx2, :]
    row2 = X2.shape[0]
    row2_choice = np.random.choice(row2, each, replace=True)
    X2_sample = X2[row2_choice, :]

    X_new = np.concatenate((X1_sample, X2_sample), axis=0)
    y_new = np.concatenate((np.ones(each), -1 * np.ones(each)), axis=0)

    return X_new, y_new

def neuronModel3D_perm(s, y, dim_left = 'time',penalty='l2', C=1, cv=16, repeat = 16, bootStrap = False):
    '''
    logistic regression, with cross validation, bootstrap and permutation test.
    :param s: spike data, (neuron * trial * time).
    :param y: label for each trial
    :param dim_left: indicate the colum in design matrix generated in the function. 'time' or 'neuron'.
    :param penalty:the penalty term used for logistic regression. 'none','l1',or'l2'.
    :param C: needed when there is penalty.
    :param cv: folds of cross validation
    :param repeat: folds of permutation. i.e., how many time scramble the data.
    :param bootStrap: whether do the boot strap for the design matrix. bool type.
    :return: ori_acc, np.mean(ori_cvd), scrambled_acc_collect, scrambled_cvd_collect [scalar, scalar, array, array]
    '''
    print('s shape =', s.shape)

    if dim_left == 'time':
      X = np.average(s, 0)
      print('X dim (time left):', X.shape)
    elif dim_left == 'neuron':
      X = np.average(s, 2)
      X = np.transpose(X)
      print('X dim (neuron left):', X.shape)
    else:
      print('Please enter "time" or "neuron" for the parameter: dim_left (set as "time" by default)')

    # add 1s line
    dim = X.shape[0]
    X = np.concatenate((np.ones(dim).reshape(dim,1),X), axis=1)
    print('add "1"s ->',X.shape)


    # boot strap
    if bootStrap:
      X, y = boot_strap(X, y)

      print('\n', 'after boot_strap, X ->', X.shape)

    # fit GLM with regularization
    if penalty == 'l2':
        model = LogisticRegression(penalty='l2',C=C,max_iter=5000)
    elif penalty == 'l1':
        model = LogisticRegression(penalty='l1',C=C,max_iter=5000,solver='saga')
    else:
        model = LogisticRegression(penalty='none', max_iter=5000)

    model.fit(X,y)

    ori_acc = compute_acc(X, y, model) # a single number
    ori_cvd = cross_val_score(model, X, y, cv=cv) # an array (cv,)

    # do permutation
    scrambled_acc_collect = np.zeros(repeat)
    scrambled_cvd_collect = np.zeros(shape=(repeat, cv))

    for i in range(repeat):
        print('running epoch', str(i))
        per = np.random.permutation(X.shape[0])
        X_scrambled = X[per,:]
        model.fit(X_scrambled, y)

        scrambled_acc = compute_acc(X_scrambled, y, model)
        scrambled_acc_collect[i] = scrambled_acc

        scrambled_cvd = cross_val_score(model, X_scrambled, y, cv=cv)
        scrambled_cvd_collect[i,:] = scrambled_cvd

    print('finished')

    return ori_acc, np.mean(ori_cvd), scrambled_acc_collect, scrambled_cvd_collect


# plot function for pca_3
def plot_3d(X1, X2, cmap1='winter', cmap2='copper', label1=None, label2=None):
    '''
    plot 3-D figure with gradient ramp color.
    :param X1: transformed matrix, dim=2. time * coordinate tick (3)
    :param X2: transformed matrix, dim=2. time * coordinate tick (3)
    :param cmap1: name of color map for X1
    :param cmap2: name of color map for X2
    :param label1: label in the plot to generate legend, for X1
    :param label2: label in the plot to generate legend, for X2
    :return: none
    '''
    fig = plt.figure(figsize=(9, 6))
    ax = Axes3D(fig)

    cm1 = plt.get_cmap(cmap1)
    cm2 = plt.get_cmap(cmap2)

    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=X1.shape[0])
    scalarMap1 = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm1)
    scalarMap2 = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm2)

    ax.scatter3D(X1[:, 0], X1[:, 1], X1[:, 2],
                 c=list(range(X1.shape[0])),
                 cmap=plt.cm.get_cmap(cmap1),
                 label=label1)

    ax.scatter3D(X2[:, 0], X2[:, 1], X2[:, 2],
                 c=list(range(X2.shape[0])),
                 cmap=plt.cm.get_cmap(cmap2),
                 label=label2)

    ax.legend()
    fig.colorbar(scalarMap1)
    fig.colorbar(scalarMap2)


def plot_2d_project(X1, X2, dim=(0, 1), cmap1='winter', cmap2='copper', label1=None, label2=None, title=None):
    '''
    plot 2-D figure with gradient ramp color.  just projection in 3 panels from a 3-D plot.
    :param X1: transformed matrix, dim=2. time * coordinate tick (3)
    :param X2: transformed matrix, dim=2. time * coordinate tick (3)
    :param dim: which panel do you want to project the 3D points to?
    :param cmap1: name of color map for X1
    :param cmap2: name of color map for X2
    :param label1: label in the plot to generate legend, for X1
    :param label2: label in the plot to generate legend, for X2
    :return: none
    '''

    fig = plt.figure(figsize=(12, 8))
    cm1 = plt.get_cmap(cmap1)
    cm2 = plt.get_cmap(cmap2)

    cNorm = matplotlib.colors.Normalize(vmin=0, vmax=X1.shape[0])
    scalarMap1 = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm1)
    scalarMap2 = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm2)

    plt.scatter(X1[:, dim[0]], X1[:, dim[1]],
                c=list(range(X1.shape[0])),
                cmap=plt.cm.get_cmap(cmap1),
                label=label1)

    plt.scatter(X2[:, dim[0]], X2[:, dim[1]],
                c=list(range(X2.shape[0])),
                cmap=plt.cm.get_cmap(cmap2),
                label=label2)

    # plt.legend()
    plt.title(title)
    fig.colorbar(scalarMap1)
    fig.colorbar(scalarMap2)


## animation production
### don't run this module in google colab.
def anime(X, dim=(0, 1), title=None, color='blue'):
    plt.ion()
    plt.title(title)

    axis1 = X[:, dim[0]]
    axis2 = X[:, dim[1]]

    plt.axis([min(axis1)-0.1,max(axis1)+0.1,min(axis2)-0.1,max(axis2)+0.1])

    xs = axis1[:2]
    ys = axis2[:2]

    for i in range(X.shape[0] - 2):
        x_new = axis1[i + 2]
        y_new = axis2[i + 2]

        xs[0] = xs[1]
        ys[0] = ys[1]
        xs[1] = x_new
        ys[1] = y_new

        plt.scatter(xs, ys, color=color)
        plt.pause(0.1)

    plt.pause(5)
    plt.close()

def anime2(X1,X2,label1,label2,
           dim=(0, 1), title=None,
           color1='pink', color2 = 'purple'):
    plt.ion()
    plt.title(title)

    X1_x = X1[:, dim[0]]
    X1_y = X1[:, dim[1]]

    X2_x = X2[:, dim[0]]
    X2_y = X2[:, dim[1]]

    minx = min(min(X1_x),min(X2_x))
    maxx = max(max(X1_x),max(X2_x))
    miny = min(min(X1_y),min(X2_y))
    maxy = max(max(X1_y),max(X2_y))

    plt.axis([minx-0.1,maxx+0.1,miny-0.1,maxy+0.1])

    x1s = X1_x[:2]
    y1s = X1_y[:2]

    x2s = X2_x[:2]
    y2s = X2_y[:2]

    for i in range(X1.shape[0] - 2):
        x1_new = X1_x[i + 2]
        y1_new = X1_y[i + 2]
        x2_new = X2_x[i + 2]
        y2_new = X2_y[i + 2]

        x1s[0] = x1s[1]
        y1s[0] = y1s[1]
        x1s[1] = x1_new
        y1s[1] = y1_new

        x2s[0] = x2s[1]
        y2s[0] = y2s[1]
        x2s[1] = x2_new
        y2s[1] = y2_new

        plt.scatter(x1s, y1s, color=color1, label=label1)
        plt.scatter(x2s, y2s, color=color2, label=label2)

        if i==0:
            plt.legend()
        plt.pause(0.1)

    plt.pause(5)
    plt.close()