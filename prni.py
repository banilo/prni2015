
"""
HCP: supervised network decomposition by articial neural networks

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
from nilearn.input_data import NiftiMasker
import theano
import theano.tensor as T
print('Running THEANO on %s' % theano.config.device)
from nilearn.image import concat_imgs
import joblib

import dnn

WRITE_DIR = op.join(os.getcwd(), '10comps')
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

##############################################################################
# load+preprocess data
##############################################################################

# mask_img = nib.load('grey10_icbm_5mm_bin.nii.gz')
mask_img='/git/dl_nets/grey10_icbm_10mm_ero2_bin.nii.gz'

# mask_img = nib.load('grey10_icbm.nii')
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()

print('Loading data...')

# X, labels, subs = joblib.load('/git/dl_nets/preload_HT')
X, labels, subs = joblib.load('/git/dl_nets/preload_HT_10mm_ero2')

# X1 = joblib.load('/git/cohort/archi/preload_1st_HT')
# labels1 = joblib.load('/git/cohort/archi/preload_1st_HT_labels')
# subs1 = joblib.load('/git/cohort/archi/preload_1st_HT_subs')
# X2, labels2, subs2 = joblib.load('/git/cohort/archi/preload_2nd_HT')
# 
# X = concat_imgs([X1, X2], accept_4d=True)
# del X1; del X2
# X = nifti_masker.transform(X)
# labels = list(labels1) + list(labels2)
# subs = list(subs1) + list(subs2)

print('done :)')

sub_ids = np.unique(subs)
train_inds = np.in1d(subs, sub_ids[:450])
test_inds = np.in1d(subs, sub_ids[450:])
# sanity-check separation of subjects
assert(not np.any(np.in1d(subs[train_inds], sub_ids[450:])))
assert(not np.any(np.in1d(subs[test_inds], sub_ids[:450])))
train_set_x, train_set_y = X[train_inds], labels[train_inds]
test_set_x, test_set_y = X[test_inds], labels[test_inds]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_set_x = ss.fit_transform(train_set_x).astype(np.float32)
test_set_x = ss.fit_transform(test_set_x).astype(np.float32)

train_set_y = train_set_y.astype(np.int32)
test_set_y = test_set_y.astype(np.int32)


##############################################################################
# compute results
##############################################################################

def dump_comps(masker, compressor, components, threshold=2):
    from scipy.stats import zscore
    from nilearn.plotting import plot_stat_map

    if isinstance(compressor, basestring):
        comp_name = compressor
    elif compressor is None:
        comp_name = 'NeuralNet'
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

n_input = X.shape[1]
n_output = 18  # number of labels
numpy_rng = np.random.RandomState(42)

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

##############################################################################
# plot figures
##############################################################################

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