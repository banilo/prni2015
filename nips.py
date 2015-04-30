"""
HCP: semi-supervised network decomposition by low-rank logistic regression

Danilo Bzdok, 2015
danilobzdok@gmail.com
"""

print __doc__

import os
import os.path as op
import numpy as np
from scipy.linalg import norm
import nibabel as nib
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
import theano
import theano.tensor as T
print('Running THEANO on %s' % theano.config.device)
from nilearn.image import concat_imgs
import joblib
import time

WRITE_DIR = op.join(os.getcwd(), 'nips')
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

##############################################################################
# load+preprocess data
##############################################################################

mask_img = 'grey10_icbm_10mm_ero2_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()

print('Loading data...')

X_task, labels, subs = joblib.load('/git/dl_nets/preload_HT_10mm_ero2')

# contrasts are IN ORDER -> shuffle!
new_inds = np.arange(0, X_task.shape[0])
np.random.shuffle(new_inds)
X_task = X_task[new_inds]
labels = labels[new_inds]
subs = subs[new_inds]

X_rest = nifti_masker.transform(
    '/git/cohort/archi/preload_HR20persub_10mm_ero2.nii')
    
X_task = StandardScaler().fit_transform(X_task)
X_rest = StandardScaler().fit_transform(X_rest)

print('done :)')

##############################################################################
# compute results
##############################################################################

class SSEncoder(BaseEstimator):
    def __init__(self, n_hidden, gain1, learning_rate, max_epochs=100):
        self.n_hidden = n_hidden
        self.gain1 = gain1
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    # def rectify(X):
    #     return T.maximum(0., X)

    from theano.tensor.shared_randomstreams import RandomStreams

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates

    def fit(self, X, y):
        self.penalty_l1 = 0.5
        self.penalty_l2 = 0.5
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.batch_size = 100
        n_input = X.shape[1]  # sklearn-like structure
        n_output = n_input
        rng = np.random.RandomState(42)
        self.input_data = T.matrix(dtype='float32', name='input_data')

        index = T.iscalar(name='index')
        
        # V -> supervised / logistic regression
        # W -> unsupervised / auto-encoder

        W0_vals = rng.randn(n_input, self.n_hidden).astype(np.float32) * self.gain1

        self.W0s = theano.shared(W0_vals)
        self.W1s = self.W0s.T  # tied

        if not DEBUG_FLAG:
            X_train_s = theano.shared(value=np.float32(X), name='X_train_s')
            y_train_s = theano.shared(value=np.int32(y), name='y_train_s')
            n_train_samples = len(X)
        else:
            from sklearn.cross_validation import StratifiedShuffleSplit
            folder = StratifiedShuffleSplit(y, n_iter=1, test_size=0.20)
            new_trains, inds_val = iter(folder).next()
            X_new_train = X[new_trains]
            X_val = X[inds_val]
            X_train_s = theano.shared(value=np.float32(X_new_train),
                                      name='X_train_s', borrow=False)
            X_val_s = theano.shared(value=np.float32(X_val),
                                    name='X_train_s', borrow=False)
            n_train_samples = len(X_new_train)
            self.dbg_epochs_ = []
            self.dbg_acc_train_ = []
            self.dbg_acc_val_ = []

        bW0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bW0s = theano.shared(value=bW0_vals, name='bW0')
        bW1_vals = np.zeros(n_output).astype(np.float32)
        self.bW1s = theano.shared(value=bW1_vals, name='bW1')

        givens_tr = {
            self.input_data: X_train_s[
                index * self.batch_size:(index + 1) * self.batch_size]
        }

        encoding = (self.input_data.dot(self.W0s) + self.bW0s).dot(self.W1s) + self.bW1s

        # auto-encoder loss
        self.ae_loss = T.sum((self.input_data - encoding) ** 2, axis=1)

        # overall cost expression
        self.cost = (T.mean(self.ae_loss) +
                     T.mean(abs(self.W0s).sum(axis=1) * self.penalty_l1) +
                     T.mean((self.W0s ** 2).sum(axis=1) * self.penalty_l2)
        )

        # cost expression
        params = [self.W0s, self.bW0s, self.bW1s]
        gparams = [T.grad(cost=self.cost, wrt=param) for param in params]

        lr = self.learning_rate
        updates = self.RMSprop(cost=self.cost, params=params,
                               lr=self.learning_rate)

        f_train = theano.function(
            [index],
            [self.cost],
            givens=givens_tr,
            updates=updates)

        self.cost_history_ = []
        start_time = time.time()
        last_cost = np.inf
        no_improve_steps = 0
        for i_epoch in range(self.max_epochs):
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))
            epoch_costs = []
            for i in range(n_train_samples // self.batch_size):
                cur_cost = f_train(i)
                epoch_costs.append(cur_cost)
                self.cost_history_.append(cur_cost)
                
            # evaluate epoch cost
            if last_cost - cur_cost[0] < 0.1:
                no_improve_steps += 1
            else:
                last_cost = cur_cost
                no_improve_steps = 0
            print('E:%i, cost:%.4f, badsteps:%i' % (
                i_epoch + 1, cur_cost[0], no_improve_steps))
            if no_improve_steps > 100:
                break  # max iter reached

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self


##############################################################################
# plot figures
##############################################################################

def dump_comps(masker, compressor, components, threshold=2):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map

    if isinstance(compressor, basestring):
        comp_name = compressor
    else:
        comp_name = compressor.__str__().split('(')[0]

    for i_c, comp in enumerate(components):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (comp_name,
                                                     n_comp, i_c + 1))
        nii_raw = masker.inverse_transform(comp)
        nii_raw.to_filename(path_mask + '.nii.gz')

        nii_z = masker.inverse_transform(zscore(comp))
        gz_path = path_mask + '_zmap.nii.gz'
        nii_z.to_filename(gz_path)
        plot_stat_map(gz_path, bg_img='colin.nii', threshold=threshold,
                      cut_coords=(0, -2, 0), draw_cross=False,
                      output_file=path_mask + 'zmap.png')

n_comp = 5
estimator = SSEncoder(
    n_hidden=n_comp,
    gain1=0.004,  # empirically determined by CV
    learning_rate = np.float32(0.00001),  # empirically determined by CV,
    max_epochs=5000)

estimator.fit(X_task, labels)
comps = estimator.W0s.get_value().T

dump_comps(nifti_masker, 'AE_n5', comps, threshold=0.1)

stop




import dnn
# dnn.add_fit_and_score(dnn.DropoutNet)
dnn.add_fit_and_score(dnn.RegularizedNet)

my_title = 'n40 logistic (rmsprop): bydropping'
fname = my_title.lower().replace(':', '').replace(' ', '_')
fpath = op.join(WRITE_DIR, fname)

test_accs = []

import matplotlib
from matplotlib import pylab as plt

accs_NN = []
accs_ICA = []
accs_PCA = []
accs_SPCA = []

n_comps = [10]
# n_comps = [1, 5] + list(np.arange(10, 101)[::10])
for n_comp in n_comps:
    print('#' * 80)
    print('#components: %i' % n_comp)
    
    clf = dnn.RegularizedNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[dnn.Linear, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        L1_reg=0.1, L2_reg=0.1,
        # dropout_rates=[float32(0.2), float32(0.5],
        debugprint=1)
    
    clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc_NN = 1 - clf.score(test_set_x, test_set_y)
    accs_NN.append(acc_NN)
    print('NN: %.4f' % acc_NN)
    first_hidden_layer = clf.params[0].get_value().T
    dump_comps(nifti_masker, None, first_hidden_layer, 0.66)


    # 2-step approach: PCA
    from sklearn.cross_validation import train_test_split
    from sklearn.linear_model import LogisticRegression
    half1_X, half2_X, half1_y, half2_y = train_test_split(
        train_set_x, train_set_y, test_size=0.5)
        
    from sklearn.decomposition import PCA
    compressor = PCA(n_components=n_comp, whiten=True)
    compressor.fit(half1_X)
    half2compr = compressor.transform(half2_X)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', max_iter=250, solver='newton-cg',
                            multi_class='multinomial')
    lr.fit(half2compr, half2_y)
    y_pred = lr.predict(compressor.transform(test_set_x))
    acc_PCA = np.mean(y_pred == test_set_y)
    print('PCA: %.4f' % acc_PCA)
    accs_PCA.append(acc_PCA)
    dump_comps(nifti_masker, compressor, compressor.components_, 0.66)
    
    # 2-step approach: ICA
    half1_X, half2_X, half1_y, half2_y = train_test_split(
        train_set_x, train_set_y, test_size=0.5)
        
    from sklearn.decomposition import FastICA
    compressor = FastICA(n_components=n_comp, whiten=True)
    compressor.fit(half1_X)
    half2compr = compressor.transform(half2_X)
    
    lr = LogisticRegression(penalty='l1', max_iter=250, solver='newton-cg',
                            multi_class='multinomial')
    lr.fit(half2compr, half2_y)
    y_pred = lr.predict(compressor.transform(test_set_x))
    acc_ICA = np.mean(y_pred == test_set_y)
    print('ICA: %.4f' % acc_ICA)
    accs_ICA.append(acc_ICA)
    dump_comps(nifti_masker, compressor, compressor.components_, 0.66)

    # 2-step approach: Sparse PCA
    half1_X, half2_X, half1_y, half2_y = train_test_split(
        train_set_x, train_set_y, test_size=0.5)
        
    from sklearn.decomposition import SparsePCA
    compressor = SparsePCA(n_components=n_comp, alpha=1.0,
        n_jobs=1, verbose=0, tol=0.1)
    compressor.fit(half1_X)
    half2compr = compressor.transform(half2_X)
    
    lr = LogisticRegression(penalty='l1', max_iter=250, solver='newton-cg',
                            multi_class='multinomial')
    lr.fit(half2compr, half2_y)
    y_pred = lr.predict(compressor.transform(test_set_x))
    acc_SPCA = np.mean(y_pred == test_set_y)
    print('SPCA: %.4f' % acc_SPCA)
    accs_SPCA.append(acc_SPCA)
    dump_comps(nifti_masker, compressor, compressor.components_, 0.66)

joblib.dump(n_comps, 'n_comps2')
joblib.dump(accs_NN, 'accs2_NN_5mm')
joblib.dump(accs_PCA, 'accs2_PCA_5mm')
joblib.dump(accs_ICA, 'accs2_ICA_5mm')
joblib.dump(accs_SPCA, 'accs2_SPCA_5mm')

n_comps = joblib.load('n_comps2')
scores_NN = joblib.load('accs2_NN_5mm')
scores_SPCA = joblib.load('accs2_SPCA_5mm')
scores_PCA = joblib.load('accs2_PCA_5mm')
scores_ICA = joblib.load('accs2_ICA_5mm')

FONTS = 12
colors = [(1., 4., 64.), (7., 136., 217.), (7., 176., 242.),
          (242., 62., 22.)]
my_colors = [(x/256, y/256, z/256) for x, y, z in colors]
plt.figure(facecolor='white', figsize=(8, 6))
b_width = 0.5
plt.xticks((b_width * 2) + (np.arange(0, 12) * (4 * b_width)),
           n_comps)
plt.yticks(np.linspace(0, 1, 11))
plt.ylim(0., 1.)
# plt.title("Integrated versus serial decomposition and classification",
#           fontsize=16)
# plt.title("HCP: Integrated versus serial decomposition and classification\n"
#           "13657 voxels per task map at 5mm isotropic", fontsize=16)
plt.xlabel('#components', fontsize=FONTS)
plt.ylabel('classification accuracy of 18 tasks', fontsize=FONTS)

cur_bin = 0
for bin in np.arange(len(n_comps)):
    if cur_bin < b_width * 4:
        plt.bar(cur_bin, scores_NN[bin], width=b_width, color=my_colors[-1],
                label='Low-rank logistic')
        cur_bin += b_width
        
        plt.bar(cur_bin, scores_PCA[bin], width=b_width, color=my_colors[0],
                label='PCA')
        cur_bin += b_width

        plt.bar(cur_bin, scores_ICA[bin], width=b_width, color=my_colors[1],
                label='FastICA')
        cur_bin += b_width

        plt.bar(cur_bin, scores_SPCA[bin], width=b_width, color=my_colors[2],
                label='SparsePCA')
        cur_bin += b_width

    else:
        plt.bar(cur_bin, scores_NN[bin], width=b_width, color=my_colors[-1])
        cur_bin += b_width

        plt.bar(cur_bin, scores_PCA[bin], width=b_width, color=my_colors[0])
        cur_bin += b_width

        plt.bar(cur_bin, scores_ICA[bin], width=b_width, color=my_colors[1])
        cur_bin += b_width

        plt.bar(cur_bin, scores_SPCA[bin], width=b_width, color=my_colors[2])
        cur_bin += b_width
        
    cur_bin += b_width

plt.legend(loc='lower right', fontsize=12)
plt.savefig('10_nn_versus_decomp_10mm_2.png', DPI=200)
plt.tight_layout()
plt.savefig('10_nn_versus_decomp_10mm_2.png', DPI=200)

stoppy

# for n_comp in n_comps:
#     print('#' * 80)
#     print('#components: %i' % n_comp)
#     
#     Neural networks
#     clf = dnn.DropoutNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
#         layers_types=[dnn.ReLU, dnn.LogisticRegression],
#         layers_sizes=[n_comp],
#         dropout_rates=[0., 0.],
#         debugprint=1)
#     
#     clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
#             verbose=False, method='rmsprop', early_stopping=True)
#     acc_NN = 1 - clf.score(test_set_x, test_set_y)
#     accs_NN.append(acc_NN)
#     print('NN: %.4f' % acc_NN)
#     first_hidden_layer = clf.params[0].get_value().T
#     dump_comps(nifti_masker, None, first_hidden_layer)
    
##############################################################################
# plot special cases
##############################################################################

# ['REWARD-PUNISH', 0
#  'PUNISH-REWARD',
#  'SHAPES-FACES',
#  'FACES-SHAPES',
#  'RANDOM-TOM',
#  'TOM-RANDOM', 5
#  'MATH-STORY',
#  'STORY-MATH',
#  'T-AVG',
#  'F-H',
#  'H-F', 10
#  'MATCH-REL',
#  'REL-MATCH',
#  'BODY-AVG',
#  'FACE-AVG', 14
#  'PLACE-AVG',
#  'TOOL-AVG', 16
#  '2BK-0BK']

for n_comp in [5, 10]:
    clf = dnn.RegularizedNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[dnn.Linear, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        L1_reg=0.1, L2_reg=0.1,
        debugprint=1)
        
    mask1 = np.logical_or(train_set_y==16, train_set_y==14)
    mask2 = np.logical_or(test_set_y==16, test_set_y==14)

    clf.fit(x_train=train_set_x[mask1], y_train=train_set_y[mask1],
            max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc_NN = 1 - clf.score(test_set_x[mask2], test_set_y[mask2])
    print('NN: %.4f' % acc_NN)
    first_hidden_layer = clf.params[0].get_value().T
    dump_comps(nifti_masker, 'face--tool', first_hidden_layer, 0.5)

##############################################################################
# plot renderings with special cuts
##############################################################################

plot_stat_map('NeuralNet_5-1_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(-51, -68, 42), draw_cross=False, output_file='NeuralNet_5-1_zmap_tar.png')
plot_stat_map('NeuralNet_5-2_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(54, -55, -6), draw_cross=False, output_file='NeuralNet_5-2_zmap_tar.png')
plot_stat_map('NeuralNet_5-3_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(0, 1, 55), draw_cross=False, output_file='NeuralNet_5-3_zmap_tar.png')
plot_stat_map('NeuralNet_5-4_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(14, -83, -6), draw_cross=False, output_file='NeuralNet_5-4_zmap_tar.png')
plot_stat_map('NeuralNet_5-5_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(0, -53, -17), draw_cross=False, output_file='NeuralNet_5-5_zmap_tar.png')

plot_stat_map('PCA_5-1_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(0, 1, 55), draw_cross=False, output_file='PCA_5-1_zmap_tar.png')
plot_stat_map('PCA_5-2_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(0, -53, -17), draw_cross=False, output_file='PCA_5-2_zmap_tar.png')
plot_stat_map('PCA_5-3_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(54, -55, -6), draw_cross=False, output_file='PCA_5-3_zmap_tar.png')
plot_stat_map('PCA_5-4_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(14, -83, -6), draw_cross=False, output_file='PCA_5-1_zmap_tar.png')
plot_stat_map('PCA_5-5_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(54, -55, -6), draw_cross=False, output_file='PCA_5-5_zmap_tar.png')

plot_stat_map('SparsePCA_5-1_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(0, 1, 55), draw_cross=False, output_file='SparsePCA_5-1_zmap_tar.png')
plot_stat_map('SparsePCA_5-2_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(-51, -68, 42), draw_cross=False, output_file='SparsePCA_5-2_zmap_tar.png')
plot_stat_map('SparsePCA_5-3_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(0, -53, -17), draw_cross=False, output_file='SparsePCA_5-3_zmap_tar.png')
plot_stat_map('SparsePCA_5-4_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(14, -83, -6), draw_cross=False, output_file='SparsePCA_5-4_zmap_tar.png')
plot_stat_map('SparsePCA_5-5_zmap.nii.gz', '../colin.nii', threshold=1, cut_coords=(54, -55, -6), draw_cross=False, output_file='SparsePCA_5-5_zmap_tar.png')


coords2 = (-46, -43, -2)
for i in range(5):
    plot_stat_map('NeuralNet_5-%i_zmap.nii.gz' % (i + 1), '../colin.nii',
                  threshold=1.5, cut_coords=(-51, -68, 42), draw_cross=False,
                  output_file='NeuralNet_5-%i_zmap_tar2.png' % (i + 1)
                  )

##############################################################################
# NN with different parameters
##############################################################################

# dropout
drs = np.linspace(0, 1, 11).astype(np.float32)
n_comp = 5
for dr in drs:
    print('#' * 80)
    print('#components: %i' % n_comp)
    print('dropout: %.2f' % dr)
    
    clf = dnn.DropoutNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[dnn.ReLU, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        dropout_rates=[dr, dr],
        debugprint=1)
    
    clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc = 1 - clf.score(test_set_x, test_set_y)
    print('accuracy: %.4f' % acc)
    first_hidden_layer = clf.params[0].get_value().T
    
    dump_comps(
        nifti_masker,
        'NeuralNet_dr%.2f_acc%.2f' % (dr, acc),
        first_hidden_layer)

# non-linearities
nonlins = ['softplus', 'tanh', 'sigmoid']
drs = np.linspace(0, 1, 11).astype(np.float32)
n_comp = 5
for nonlin in nonlins:
    print('#' * 80)
    print('#components: %i' % n_comp)
    print('non linear function: %s' % nonlin)

    if nonlin == 'softplus':
        nonlinfunc = dnn.SoftPlusLayer
    elif nonlin == 'tanh':
        nonlinfunc = dnn.TanhLayer
    elif nonlin == 'sigmoid':
        nonlinfunc = dnn.SigmoidLayer
    else:
        nonlinfunc = None

    clf = dnn.DropoutNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[nonlinfunc, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        dropout_rates=[0., 0.],
        debugprint=1)

    clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc = 1 - clf.score(test_set_x, test_set_y)
    print('accuracy: %.4f' % acc)
    first_hidden_layer = clf.params[0].get_value().T

    dump_comps(
        nifti_masker,
        'NeuralNet_nonlin(%s)_acc%.2f' % (nonlin, acc),
        first_hidden_layer)

# pretraining of W0 by PCA, ICA and SparsePCA
pretrains = ['FastICA', 'PCA', 'SparsePCA']
n_comp = 5
for pre in pretrains:
    print('#' * 80)
    print('#components: %i' % n_comp)
    print('pretraining: %s' % pre)

    clf = dnn.DropoutNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[dnn.Linear, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        dropout_rates=[0., 0.],
        debugprint=1)

    # hack the preset weight matrix
    knownW = np.zeros((n_comp, mask_nvox), dtype=np.float32)
    for i_c in np.arange(n_comp):
        path_mask = op.join(WRITE_DIR, '%s_%i-%i' % (pre,
                                                     n_comp, i_c + 1))
        path_mask += '.nii.gz'
        cur_comp = nifti_masker.transform(path_mask)
        knownW[i_c, :] = cur_comp
    clf.params[0].set_value(knownW.T)

    clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc = 1 - clf.score(test_set_x, test_set_y)
    print('accuracy: %.4f' % acc)
    first_hidden_layer = clf.params[0].get_value().T

    dump_comps(
        nifti_masker,
        'NeuralNet_pretrained(%s)_acc%.2f' % (pre, acc),
        first_hidden_layer)

# L1 regularization
L1s = [0.1, 0.3, 0.5]
n_comp = 5
for L1 in L1s:
    print('#' * 80)
    print('#components: %i' % n_comp)
    print('L1: %.1f' % L1)

    clf = dnn.RegularizedNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[dnn.Linear, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        L1_reg=L1, L2_reg=0,
        debugprint=1)

    clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc = 1 - clf.score(test_set_x, test_set_y)
    print('accuracy: %.4f' % acc)
    first_hidden_layer = clf.params[0].get_value().T

    dump_comps(
        nifti_masker,
        'NeuralNet_pretrained(L1=%.1f)_acc%.2f' % (L1, acc),
        first_hidden_layer)

# L2 regularization
L2s = [0.1, 0.3, 0.5]
n_comp = 5
for L2 in L2s:
    print('#' * 80)
    print('#components: %i' % n_comp)
    print('L2: %.1f' % L2)

    clf = dnn.RegularizedNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[dnn.Linear, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        L1_reg=0, L2_reg=L2,
        debugprint=1)

    clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc = 1 - clf.score(test_set_x, test_set_y)
    print('accuracy: %.4f' % acc)
    first_hidden_layer = clf.params[0].get_value().T

    dump_comps(
        nifti_masker,
        'NeuralNet_pretrained(L2=%.1f)_acc%.2f' % (L2, acc),
        first_hidden_layer)

# L1 + L2
Ls = [0.1, 0.3, 0.5]
n_comp = 5
for L in Ls:
    print('#' * 80)
    print('#components: %i' % n_comp)
    print('L1/2: %.1f' % L)

    clf = dnn.RegularizedNet(numpy_rng=numpy_rng, n_ins=n_input, n_outs=n_output,
        layers_types=[dnn.Linear, dnn.LogisticRegression],
        layers_sizes=[n_comp],
        L1_reg=L, L2_reg=L,
        debugprint=1)

    clf.fit(x_train=train_set_x, y_train=train_set_y, max_epochs=250, split_ratio=0.1,
            verbose=False, method='rmsprop', early_stopping=True)
    acc = 1 - clf.score(test_set_x, test_set_y)
    print('accuracy: %.4f' % acc)
    first_hidden_layer = clf.params[0].get_value().T

    dump_comps(
        nifti_masker,
        'NeuralNet_pretrained(L1andL2=%.1f)_acc%.2f' % (L, acc),
        first_hidden_layer)