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

labels = np.int32(labels)

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

    def fit(self, X_rest, X_task, y):
        self.penalty_l1 = 0.1
        self.penalty_l2 = 0.1
        DEBUG_FLAG = True

        # self.max_epochs = 333
        self.batch_size = 100
        n_input = X_rest.shape[1]  # sklearn-like structure
        n_output = n_input
        rng = np.random.RandomState(42)
        self.input_data = T.matrix(dtype='float32', name='input_data')

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
            X_val_s = theano.shared(value=np.float32(X_val),
                                    name='X_train_s', borrow=False)
            y_val_s = theano.shared(value=np.int32(y_val),
                                    name='y_cal_s', borrow=False)
            lr_train_samples = len(X_train)
            self.dbg_epochs_ = []
            self.dbg_acc_train_ = []
            self.dbg_acc_val_ = []
        X_rest_s = theano.shared(value=np.float32(X_rest), name='X_rest_s')
        ae_train_samples = len(X_rest)

        # V -> supervised / logistic regression
        # W -> unsupervised / auto-encoder

        # computation graph: auto-encoder
        W0_vals = rng.randn(n_input, self.n_hidden).astype(np.float32) * self.gain1

        self.W0s = theano.shared(W0_vals)
        self.W1s = self.W0s.T  # tied
        bW0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bW0s = theano.shared(value=bW0_vals, name='bW0')
        bW1_vals = np.zeros(n_output).astype(np.float32)
        self.bW1s = theano.shared(value=bW1_vals, name='bW1')

        givens_ae = {
            self.input_data: X_rest_s[
                index * self.batch_size:(index + 1) * self.batch_size]
        }

        encoding = (self.input_data.dot(self.W0s) + self.bW0s).dot(self.W1s) + self.bW1s

        self.ae_loss = T.sum((self.input_data - encoding) ** 2, axis=1)

        self.ae_cost = (
            T.mean(self.ae_loss)
        )

        params1 = [self.W0s, self.bW0s, self.bW1s]
        gparams1 = [T.grad(cost=self.ae_cost, wrt=param1) for param1 in params1]

        lr = self.learning_rate
        updates = self.RMSprop(cost=self.ae_cost, params=params1,
                               lr=self.learning_rate)

        f_train_ae = theano.function(
            [index],
            [self.ae_cost],
            givens=givens_ae,
            updates=updates)

        # computation graph: logistic regression
        clf_n_output = 18  # number of labels
        my_y = T.ivector(name='y')

        bV0_vals = np.zeros(self.n_hidden).astype(np.float32)
        self.bV0 = theano.shared(value=bV0_vals, name='bV0')
        bV1_vals = np.zeros(clf_n_output).astype(np.float32)
        self.bV1 = theano.shared(value=bV1_vals, name='bV1')
        
        V1_vals = rng.randn(self.n_hidden, clf_n_output).astype(np.float32) * self.gain1
        self.V1s = theano.shared(V1_vals)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(T.dot(self.input_data, self.W0s) + self.bV0, self.V1s) + self.bV1
        )
        self.lr_cost = -T.mean(T.log(self.p_y_given_x)[T.arange(my_y.shape[0]), my_y])
        self.lr_cost = (
            self.lr_cost +
            T.mean(abs(self.W0s).sum(axis=1) * self.penalty_l1) +
            T.mean(abs(self.bV0) * self.penalty_l1) +
            T.mean(abs(self.V1s).sum(axis=1) * self.penalty_l1) +
            T.mean(abs(self.bV1) * self.penalty_l1) +
            T.mean((self.W0s ** 2).sum(axis=1) * self.penalty_l2) +
            T.mean((self.bV0 ** 2) * self.penalty_l2) +
            T.mean((self.V1s ** 2).sum(axis=1) * self.penalty_l2) +
            T.mean((self.bV1 ** 2) * self.penalty_l2)
        )
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        givens_lr = {
            self.input_data: X_train_s[index * self.batch_size:(index + 1) * self.batch_size],
            my_y: y_train_s[index * self.batch_size:(index + 1) * self.batch_size]
        }

        params2 = [self.W0s, self.bV0, self.V1s, self.bV1]
        updates2 = self.RMSprop(cost=self.lr_cost, params=params2,
                                lr=self.learning_rate)

        # gparams2 = [T.grad(cost=self.lr_cost, wrt=param2) for param2 in params2]
        # updates2 = [
        #     (param2, param2 - lr * gparam2)
        #     for param2, gparam2 in zip(params2, gparams2)]

        f_train_lr = theano.function(
            [index],
            [self.lr_cost],
            givens=givens_lr,
            updates=updates2)

        # optimization loop
        self.ae_cost_history_ = []
        start_time = time.time()
        ae_last_cost = np.inf
        lr_last_cost = np.inf
        no_improve_steps = 0
        acc_train, acc_val = 0., 0.
        for i_epoch in range(self.max_epochs):
            if i_epoch == 1:
                epoch_dur = time.time() - start_time
                total_mins = (epoch_dur * self.max_epochs) / 60
                hs, mins = divmod(total_mins, 60)
                print("Max estimated duration: %i hours and %i minutes" % (hs, mins))
            ae_epoch_costs = []

            # if i_epoch % 2 == 0:  # every second time
            #if False:
                # auto-encoder
            # for i in range(ae_train_samples // self.batch_size):
            #         ae_cur_cost = f_train_ae(i)
            # # evaluate epoch cost
            # if ae_last_cost - ae_cur_cost[0] < 0.1:
            #     no_improve_steps += 1
            # else:
            #     ae_last_cost = ae_cur_cost[0]
            #     no_improve_steps = 0

            # logistic
            for i in range(lr_train_samples // self.batch_size):
                    lr_cur_cost = f_train_lr(i)
            lr_last_cost = lr_cur_cost[0]
            acc_train = self.score(X_train, y_train)
            acc_val = self.score(X_val, y_val)

            print('E:%i, ae_cost:%.4f, lr_cost:%.4f, train_score:%.2f, vald_score:%.2f, ae_badsteps:%i' % (
                i_epoch + 1, ae_last_cost, lr_last_cost, acc_train, acc_val, no_improve_steps))

            # ae_epoch_costs.append(ae_cur_cost)
            # self.ae_cost_history_.append(ae_cur_cost)

            # if no_improve_steps > 100:
            #     break  # max iter reached

        total_mins = (time.time() - start_time) / 60
        hs, mins = divmod(total_mins, 60)
        print("Final duration: %i hours and %i minutes" % (hs, mins))

        return self

    def predict(self, X):
        X_test_s = theano.shared(value=np.float32(X), name='X_test_s', borrow=True)

        givens_te = {
            self.input_data: X_test_s
        }

        f_test = theano.function(
            [],
            [self.y_pred],
            givens=givens_te)
        predictions = f_test()
        return predictions

    def score(self, X, y):
        acc = np.mean(self.predict(X) == y)
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

n_comp = 20
estimator = SSEncoder(
    n_hidden=n_comp,
    gain1=0.004,  # empirically determined by CV
    learning_rate = np.float32(0.00001),  # empirically determined by CV,
    max_epochs=5000)

estimator.fit(X_rest, X_task, labels)

comps = estimator.W0s.get_value().T
dump_comps(nifti_masker, 'AE_n5', comps, threshold=0.1)

stop




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