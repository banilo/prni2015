"""
HCP: semi-supervised network decomposition by low-rank logistic regression
"""
"""
"""

print __doc__

import os
import os.path as op
import numpy as np
import glob
from scipy.linalg import norm
import nibabel as nib
from sklearn.grid_search import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from nilearn.input_data import NiftiMasker
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import theano
import theano.tensor as T
from matplotlib import pylab as plt
print('Running THEANO on %s' % theano.config.device)
from nilearn.image import concat_imgs
import joblib
import time

RES_NAME = 'nips3mm_vanilla'
WRITE_DIR = op.join(os.getcwd(), RES_NAME)
if not op.exists(WRITE_DIR):
    os.mkdir(WRITE_DIR)

##############################################################################
# load+preprocess data
##############################################################################

mask_img = 'grey10_icbm_3mm_bin.nii.gz'
nifti_masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=False,
                           standardize=False)
nifti_masker.fit()
mask_nvox = nifti_masker.mask_img_.get_data().sum()

print('Loading data...')

# ARCHI task
X_task, labels = joblib.load('preload_HT_3mm')

labels = np.int32(labels)

# contrasts are IN ORDER -> shuffle!
new_inds = np.arange(0, X_task.shape[0])
np.random.shuffle(new_inds)
X_task = X_task[new_inds]
labels = labels[new_inds]
# subs = subs[new_inds]

X_task = StandardScaler().fit_transform(X_task)

# ARCHI task
AT_niis, AT_labels, AT_subs = joblib.load('preload_AT_3mm')
AT_X = nifti_masker.transform(AT_niis)
AT_X = StandardScaler().fit_transform(AT_X)
print('done :)')

##############################################################################
# define computation graph
##############################################################################

class SSEncoder(BaseEstimator):
    def __init__(self, gain1, learning_rate, max_epochs=100,
                 l1=0.1, l2=0.1):
        """
        Parameters
        ----------
        lambda : float
            Mediates between AE and LR. lambda==1 equates with LR only.
        """
        self.gain1 = gain1
        self.max_epochs = max_epochs
        self.learning_rate = np.float32(learning_rate)
        self.penalty_l1 = np.float32(l1)
        self.penalty_l2 = np.float32(l2)

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
        
    def get_param_pool(self):
        cur_params = (
            self.W0s, self.bW0s
        )
        return cur_params
        
    def test_performance_in_other_dataset(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.cross_validation import StratifiedShuffleSplit

        compr_matrix = self.W0s.get_value().T  # currently best compression
        AT_X_compr = np.dot(compr_matrix, AT_X.T).T
        clf = LogisticRegression(penalty='l1')
        folder = StratifiedShuffleSplit(y=AT_labels, n_iter=5, test_size=0.2)

        acc_list = []
        prfs_list = []
        for (train_inds, test_inds) in folder:
            clf.fit(AT_X_compr[train_inds, :], AT_labels[train_inds])
            pred_y = clf.predict(AT_X_compr[test_inds, :])

            acc = (pred_y == AT_labels[test_inds]).mean()
            prfs_list.append(precision_recall_fscore_support(
                             AT_labels[test_inds], pred_y))

            acc_list.append(acc)

        compr_mean_acc = np.mean(acc_list)
        prfs = np.asarray(prfs_list).mean(axis=0)
        return compr_mean_acc, prfs

    def fit(self, X_task, y):
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.batch_size = 100
        rng = np.random.RandomState(42)
        self.input_taskdata = T.matrix(dtype='float32', name='input_taskdata')
        self.input_restdata = T.matrix(dtype='float32', name='input_restdata')
        self.params_from_last_iters = []

        index = T.iscalar(name='index')
        
        # prepare data for theano computation
        if not DEBUG_FLAG:
            X_train_s = theano.shared(
                value=np.float32(X_task), name='X_train_s')
            y_train_s = theano.shared(
                value=np.int32(y), name='y_train_s')
            lr_train_samples = len(X_task)
        else:
            from sklearn.cross_validation import StratifiedShuffleSplit
            folder = StratifiedShuffleSplit(y, n_iter=1, test_size=0.20)
            new_trains, inds_val = iter(folder).next()
            X_train, X_val = X_task[new_trains], X_task[inds_val]
            y_train, y_val = y[new_trains], y[inds_val]

            X_train_s = theano.shared(value=np.float32(X_train),
                                      name='X_train_s', borrow=False)
            y_train_s = theano.shared(value=np.int32(y_train),
                                      name='y_train_s', borrow=False)
            # X_val_s = theano.shared(value=np.float32(X_val),
            #                         name='X_train_s', borrow=False)
            # y_val_s = theano.shared(value=np.int32(y_val),
            #                         name='y_cal_s', borrow=False)
            lr_train_samples = len(X_train)
            self.dbg_epochs_ = list()
            self.dbg_acc_train_ = list()
            self.dbg_acc_val_ = list()
            self.dbg_ae_cost_ = list()
            self.dbg_lr_cost_ = list()
            self.dbg_ae_nonimprovesteps = list()
            self.dbg_acc_other_ds_ = list()
            self.dbg_combined_cost_ = list()
            self.dbg_prfs_ = list()
            self.dbg_prfs_other_ds_ = list()

        # computation graph: logistic regression
        clf_n_output = 18  # number of labels
        my_y = T.ivector(name='y')

        bV0_vals = np.zeros(clf_n_output).astype(np.float32)
        self.bV0 = theano.shared(value=bV0_vals, name='bV0')

        V0_vals = rng.randn(n_input, clf_n_output).astype(np.float32) * self.gain1
        self.V0s = theano.shared(V0_vals)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(self.input_taskdata, self.V0s) + self.bV0
        )
        self.lr_cost = -T.mean(T.log(self.p_y_given_x)[T.arange(my_y.shape[0]), my_y])
        self.lr_cost = (
            self.lr_cost +
            T.mean(abs(self.V0s)) * self.penalty_l1 +
            T.mean(abs(self.bV0)) * self.penalty_l1 +

            T.mean((self.V0s ** np.float32(2))) * self.penalty_l2 +
            T.mean((self.bV0 ** np.float32(2))) * self.penalty_l2
        )
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        givens_lr = {
            self.input_taskdata: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }

        params = [self.V0s, self.bV0]
        updates = self.RMSprop(cost=self.lr_cost, params=params,
                               lr=self.learning_rate)

        f_train_lr = theano.function(
            [index],
            [self.lr_cost],
            givens=givens_lr,
            updates=updates)

        # optimization loop
        start_time = time.time()
        lr_last_cost = np.inf
        no_improve_steps = 0
        acc_train, acc_val = 0., 0.
        for i_epoch in range(self.max_epochs):
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))

            lr_n_batches = lr_train_samples // self.batch_size
            for i in range(lr_n_batches):
                lr_cur_cost = f_train_lr(i)

            # evaluate epoch cost
            if lr_last_cost - lr_cur_cost < 0.1:
                no_improve_steps += 1
            else:
                lr_last_cost = lr_cur_cost
                no_improve_steps = 0

            # logistic
            lr_last_cost = lr_cur_cost
            acc_train = self.score(X_train, y_train)
            acc_val, prfs_val = self.score(X_val, y_val, return_prfs=True)

            print('E:%i, ae_cost:%.4f, lr_cost:%.4f, train_score:%.2f, vald_score:%.2f, ae_badsteps:%i' % (
                i_epoch + 1, ae_cur_cost, lr_cur_cost, acc_train, acc_val, no_improve_steps))

            if (i_epoch % 10 == 0):
                self.dbg_ae_cost_.append(ae_cur_cost)
                self.dbg_lr_cost_.append(lr_cur_cost)
                self.dbg_combined_cost_.append(combined_cost)

                self.dbg_epochs_.append(i_epoch + 1)
                self.dbg_ae_nonimprovesteps.append(no_improve_steps)
                self.dbg_acc_train_.append(acc_train)
                self.dbg_acc_val_.append(acc_val)
                self.dbg_prfs_.append(prfs_val)

                # test out-of-dataset performance
                od_acc, prfs_other = self.test_performance_in_other_dataset()
                self.dbg_acc_other_ds_.append(od_acc)
                self.dbg_prfs_other_ds_.append(prfs_other)
                print('out-of-dataset acc: %.2f' % od_acc)
                
            # if i_epoch > (self.max_epochs - 100):
            param_pool = self.get_param_pool()
            self.params_from_last_iters.append(param_pool)

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self

    def predict(self, X):
        X_test_s = theano.shared(value=np.float32(X), name='X_test_s', borrow=True)

        givens_te = {
            self.input_taskdata: X_test_s
        }

        f_test = theano.function(
            [],
            [self.y_pred],
            givens=givens_te)
        predictions = f_test()
        del X_test_s
        del givens_te
        return predictions[0]

    def score(self, X, y, return_prfs=False):
        pred_y = self.predict(X)
        acc = np.mean(pred_y == y)
        prfs = precision_recall_fscore_support(pred_y, y)
        if return_prfs:
            return acc, prfs
        else:
            return acc


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


l1 = 0.1
l2 = 0.1
my_title = r'LR: L1=%.1f L2=%.1f res=3mm' % (
    l1, l2
)
print(my_title)
estimator = SSEncoder(
    gain1=0.004,  # empirically determined by CV
    learning_rate = np.float32(0.00001),  # empirically determined by CV,
    max_epochs=500, l1=l1, l2=l2)

estimator.fit(X_task, labels)

# my_title = r'Low-rank LR: n_comp=%i L1=%.1f L2=%.1f res=10mm pca20RS' % (
#     n_comp, l1, l2
# )
# FONTS = 12
# plt.figure(facecolor='white', figsize=(8, 6))
# plt.plot(np.log(estimator.dbg_ae_cost_), label='cost autoencoder')
# plt.plot(estimator.dbg_lr_cost_, label='cost logistic')
# plt.plot(estimator.dbg_acc_train_, label='score training set')
# plt.plot(estimator.dbg_acc_val_, label='score validation set')
# plt.plot(estimator.dbg_acc_other_ds_, label='other-datset acc')
# plt.legend(loc='best', fontsize=12)
# plt.xlabel('epoch', fontsize=FONTS)
# plt.ylabel('misc', fontsize=FONTS)
# plt.yticks(np.arange(12), np.arange(12))
# plt.grid(True)
# plt.title(my_title)
# plt.show()

fname = my_title.replace(' ', '_').replace('+', '').replace(':', '').replace('__', '_').replace('%', '')
cur_path = op.join(WRITE_DIR, fname)
joblib.dump(estimator, cur_path)
# plt.savefig(cur_path + '_SUMMARY.png', dpi=200)

# dump data also as numpy array
np.save(cur_path + 'dbg_epochs_', np.array(estimator.dbg_epochs_))
np.save(cur_path + 'dbg_acc_train_', np.array(estimator.dbg_acc_train_))
np.save(cur_path + 'dbg_acc_val_', np.array(estimator.dbg_acc_val_))
np.save(cur_path + 'dbg_ae_cost_', np.array(estimator.dbg_ae_cost_))
np.save(cur_path + 'dbg_lr_cost_', np.array(estimator.dbg_lr_cost_))
np.save(cur_path + 'dbg_ae_nonimprovesteps', np.array(estimator.dbg_ae_nonimprovesteps))
np.save(cur_path + 'dbg_acc_other_ds_', np.array(estimator.dbg_acc_other_ds_))
np.save(cur_path + 'dbg_combined_cost_', np.array(estimator.dbg_combined_cost_))
np.save(cur_path + 'dbg_prfs_', np.array(estimator.dbg_prfs_))
np.save(cur_path + 'dbg_prfs_other_ds_', np.array(estimator.dbg_prfs_other_ds_))

V0_mat = estimator.V0s.get_value().T
np.save(cur_path + 'V0comps', V0_mat)

V1_mat = estimator.V1s.get_value().T
np.save(cur_path + 'V1comps', V1_mat)
dump_comps(nifti_masker, fname, comps, threshold=0.25)

# equally scaled plots
# import re
# pkgs = glob.glob(RES_NAME + '/*dbg_epochs*.npy')
# dbg_epochs_ = np.load(pkgs[0])
# n_comps = [100]
# pkgs = glob.glob(RES_NAME + '/*dbg_acc_train*.npy')
# for n_comp in n_comps:
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         # n_hidden = int(re.search('comp=(?P<comp>.{1,2,3})_', p).group('comp'))
#         n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_acc_train_ = np.load(p)
#         
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'spca20RS' in p:
#             cur_label += 'RSspca20'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         cur_label += '' if '_AE' in p else '/LR only!'
#         plt.plot(
#             dbg_epochs_,
#             dbg_acc_train_,
#             label=cur_label)
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
#     plt.legend(loc='lower right', fontsize=9)
#     plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('training accuracy')
#     plt.xlabel('epochs')
#     plt.ylim(0., 1.05)
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'accuracy_train_%icomps.png' % n_comp))
# 
# pkgs = glob.glob(RES_NAME + '/*dbg_acc_val_*.npy')
# for n_comp in n_comps:  # 
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         # n_hidden = int(re.search('comp=(?P<comp>.{1,2,3})_', p).group('comp'))
#         n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_acc_val_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         plt.plot(
#             dbg_epochs_,
#             dbg_acc_val_,
#             label=cur_label)
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
#     plt.legend(loc='lower right', fontsize=9)
#     plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('validation set accuracy')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'accuracy_val_%icomps.png' % n_comp))
# 
# pkgs = glob.glob(RES_NAME + '/*dbg_acc_other_ds_*.npy')
# for n_comp in n_comps:  # 
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_acc_other_ds_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         plt.plot(
#             dbg_epochs_,
#             dbg_acc_other_ds_,
#             label=cur_label)
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
#     plt.legend(loc='lower right', fontsize=9)
#     plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('ARCHI dataset accuracy')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'accuracy_archi_%icomps.png' % n_comp))
# 
# pkgs = glob.glob(RES_NAME + '/*dbg_ae_cost_*.npy')
# for n_comp in n_comps:  # AE
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_ae_cost_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         plt.plot(
#             dbg_epochs_,
#             dbg_ae_cost_,
#             label=cur_label)
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('AE loss')
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'loss_ae_%icomps.png' % n_comp))
# 
# pkgs = glob.glob(RES_NAME + '/*dbg_lr_cost_*.npy')  # LR cost
# for n_comp in n_comps:  # AE
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_lr_cost_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         plt.plot(
#             dbg_epochs_,
#             dbg_lr_cost_,
#             label=cur_label)
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('LR loss')
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'loss_lr_%icomps.png' % n_comp))
# 
# pkgs = glob.glob(RES_NAME + '/*dbg_combined_cost_*.npy')  # combined loss
# for n_comp in n_comps:  # AE
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_combined_cost_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         plt.plot(
#             dbg_epochs_,
#             dbg_combined_cost_,
#             label=cur_label)
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('combined loss')
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'loss_combined_%icomps.png' % n_comp))
# 
# pkgs = glob.glob(RES_NAME + '/*lambda=0.5*dbg_prfs_.npy')
# for n_comp in n_comps:
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_prfs_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         for i in np.arange(18):
#             plt.plot(
#                 dbg_epochs_,
#                 np.array(dbg_prfs_)[:, 0, i],
#                 label='task %i' % (i + 1))
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=0.5')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('in-dataset precisions')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'prec_inds_%icomps.png' % n_comp))
# 
# # in-dataset recall at lambda=0.5
# pkgs = glob.glob(RES_NAME + '/*lambda=0.5*dbg_prfs_.npy')
# for n_comp in n_comps:
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
#         if n_comp != n_hidden:
#             continue
#         
#         dbg_prfs_ = np.load(p)
#         
#         dbg_prfs_ = np.load(p)
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         for i in np.arange(18):
#             plt.plot(
#                 dbg_epochs_,
#                 np.array(dbg_prfs_)[:, 1, i],
#                 label='task %i' % (i + 1))
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=0.5')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('in-dataset recall')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'rec_inds_%icomps.png' % n_comp))
# 
# # in-dataset f1 at lambda=0.5
# pkgs = glob.glob(RES_NAME + '/*lambda=0.5*dbg_prfs_.npy')
# for n_comp in n_comps:
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
#         if n_comp != n_hidden:
#             continue
#             
#         dbg_prfs_ = np.load(p)
#             
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         for i in np.arange(18):
#             plt.plot(
#                 dbg_epochs_,
#                 np.array(dbg_prfs_)[:, 2, i],
#                 label='task %i' % (i + 1))
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=0.5')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('in-dataset f1 score')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'f1_inds_%icomps.png' % n_comp))
# 
# # out-of-dataset precision at lambda=0.5
# pkgs = glob.glob(RES_NAME + '/*lambda=0.5*dbg_prfs_other_ds_.npy')
# for n_comp in n_comps:
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
#         if n_comp != n_hidden:
#             continue
#             
#         dbg_prfs_other_ds_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         for i in np.arange(18):
#             plt.plot(
#                 dbg_epochs_,
#                 np.array(dbg_prfs_other_ds_)[:, 0, i],
#                 label='task %i' % (i + 1))
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=0.5')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('out-of-dataset precisions')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'prec_oods_%icomps.png' % n_comp))
# 
# # out-of-dataset recall at lambda=0.5
# pkgs = glob.glob(RES_NAME + '/*lambda=0.5*dbg_prfs_other_ds_.npy')
# for n_comp in n_comps:
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
#         if n_comp != n_hidden:
#             continue
#             
#         dbg_prfs_other_ds_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         for i in np.arange(18):
#             plt.plot(
#                 dbg_epochs_,
#                 np.array(dbg_prfs_other_ds_)[:, 1, i],
#                 label='task %i' % (i + 1))
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=0.5')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('out-of-dataset recall')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'rec_oods_%icomps.png' % n_comp))
# 
# # out-of-dataset f1 at lambda=0.5
# pkgs = glob.glob(RES_NAME + '/*lambda=0.5*dbg_prfs_other_ds_.npy')
# for n_comp in n_comps:
#     plt.figure()
#     for p in pkgs:
#         lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#         n_hidden = int(re.search('comp=(?P<comp>.{1,3})_', p).group('comp'))
#         if n_comp != n_hidden:
#             continue
#             
#         dbg_prfs_other_ds_ = np.load(p)
# 
#         cur_label = 'n_comp=%i' % n_hidden
#         cur_label += '/'
#         cur_label += 'lambda=%.2f' % lambda_param
#         cur_label += '/'
#         if not '_AE' in p:
#             cur_label += 'LR only!'
#         elif 'subRS' in p:
#             cur_label += 'RSnormal'
#         elif 'pca20RS' in p:
#             cur_label += 'RSpca20'
#         cur_label += '/'
#         cur_label += 'separate decomp.' if 'decomp_separate' in p else 'joint decomp.'
#         for i in np.arange(18):
#             plt.plot(
#                 dbg_epochs_,
#                 np.array(dbg_prfs_other_ds_)[:, 2, i],
#                 label='task %i' % (i + 1))
#     plt.title('Low-rank LR+AE L1=0.1 L2=0.1 res=3mm combined-loss lambda=0.5')
#     plt.legend(loc='lower right', fontsize=9)
#     # plt.yticks(np.linspace(0., 1., 11))
#     plt.ylabel('out-of-dataset f1 score')
#     plt.ylim(0., 1.05)
#     plt.xlabel('epochs')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(op.join(WRITE_DIR, 'f1_oods_%icomps.png' % n_comp))
# 
# # print network components (1st layer)
# pkgs = glob.glob(RES_NAME + '/*W0comps.npy')
# for p in pkgs:
#     lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#     n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#     if n_comp != n_hidden:
#         continue
#         
#     new_fname = 'comps_n=%i_lambda=%.2f_th0.5' % (n_hidden, lambda_param)
#     comps = np.load(p)
#     dump_comps(nifti_masker, new_fname, comps, threshold=0.5)
# 
# pkgs = glob.glob(RES_NAME + '/*comps*.npy')
# for p in pkgs:
#     lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#     n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#     if 'V1' in p:
#         continue
#     if n_comp != n_hidden:
#         continue
#     print p
#         
#     new_fname = 'comps_n=%i_lambda=%.2f_th0.0' % (n_hidden, lambda_param)
#     comps = np.load(p)
#     dump_comps(nifti_masker, new_fname, comps, threshold=0.0)
# 
# # print LR decision matrix (2nd layer)
# pkgs = glob.glob(RES_NAME + '/*V1comps.npy')
# for p in pkgs:
#     lambda_param = np.float(re.search('lambda=(.{4})', p).group(1))
#     n_hidden = int(re.search('comp=(.{1,3})_', p).group(1))
#     if n_comp != n_hidden:
#         continue
#     print p
#     
#     cur_mat = np.load(p)
# 
#     if n_comp == 20:
#         fs = (8, 6)
#     elif n_comp == 100:
#         fs = (12, 8)
#         
# 
#     plt.figure(figsize=fs)
#     masked_data = np.ma.masked_where(cur_mat != 0., cur_mat)
#     plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.gray_r)
#     masked_data = np.ma.masked_where(cur_mat == 0., cur_mat)
#     plt.imshow(masked_data, interpolation='nearest', cmap=plt.cm.RdBu_r)
#     plt.show()
# 
#     # plt.xticks(range(n_comp)[::5], (np.arange(n_comp) + 1)[::5])
#     # plt.xlabel('hidden components')
#     # plt.yticks(range(18), np.arange(18) + 1)
#     # plt.ylabel('tasks')
#     # plt.title('Linear combinations of component per task')
#     # plt.colorbar()
#     
#     new_fname = 'comps_n=%i_lambda=%.2f_V1_net2task.png' % (n_hidden, lambda_param)
#     plt.savefig(op.join(WRITE_DIR, new_fname))