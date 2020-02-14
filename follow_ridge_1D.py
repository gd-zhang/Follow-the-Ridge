import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import time
import random
import pickle

import tensorflow as tf
import numpy as np

from collections import OrderedDict
from tqdm import tqdm

from utils.data import MOG_1D
from utils.misc import *
from utils.optim import RMSProp
from utils.logger import get_logger

# fix random seed for np
np.random.seed(2020)
random.seed(2020)

parser = argparse.ArgumentParser()
parser.add_argument("--iteration", type=int, default=50000, help="number of iterations of training")
parser.add_argument("--batch_size", type=int, default=5000, help="size of the batches")
parser.add_argument("--gen_learning_rate", "--g_lr", type=float, default=0.0002, help="generator learning rate")
parser.add_argument("--disc_learning_rate", "--d_lr", type=float, default=0.0002, help="discriminator learning rate")
parser.add_argument("--weight_decay", "--wd", type=float, default=0.0001, help="weight decay coefficient for disc")
parser.add_argument("--z_dim", type=int, default=16, help="dimension of latent node")
parser.add_argument("--g_hidden", type=int, default=64, help="dimension of hidden units")
parser.add_argument("--d_hidden", type=int, default=64, help="dimension of hidden units")
parser.add_argument("--d_layers", type=int, default=2, help="num of hidden layer")
parser.add_argument("--g_layers", type=int, default=2, help="num of hidden layer")
parser.add_argument("--x_dim", type=int, default=1, help="data dimension")
parser.add_argument("--momentum", type=float, default=0.0, help="momentum coefficient for the whole system")
parser.add_argument("--init", type=str, default="xavier", help="which initialization scheme")
parser.add_argument("--data", type=str, default="MOG-1D", help="which dataset")
parser.add_argument("--data_std", type=float, default=0.1, help="std for each mode")
parser.add_argument("--g_act", type=str, default="tanh", help="which activation function for gen")
parser.add_argument("--d_act", type=str, default="tanh", help="which activation function for disc")

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument("--vanilla_gradient", "--vg", action='store_true', help="whether to remove preconditioning")
parser.add_argument("--follow_ridge", action='store_true', help="whether to use  follow-the-ridge")
parser.add_argument("--inner_iter", type=int, default=5, help="conjugate gradient or gradient descent steps")
parser.add_argument("--damping", type=float, default=1.0, help="initial damping term for CG")
parser.add_argument("--adapt_damping", action='store_true', help="whether to adapt damping")
opt = parser.parse_args()
print(opt)


# automatically setup the name
name = '%s-std%.2f/' % (opt.data, opt.data_std)
name += 'bs%d-z%d-g%dh%d-d%dh%d-ga%s-da%s-glr%.5f-dlr%.5f-%s-wd%.4f-mom%.2f' \
       % (opt.batch_size, opt.z_dim, opt.g_layers, opt.g_hidden, opt.d_layers, opt.d_hidden,
          opt.g_act, opt.d_act, opt.gen_learning_rate, opt.disc_learning_rate, opt.init, opt.weight_decay, opt.momentum)

if opt.follow_ridge:
    name += '-cg%d-damp%.3f' % (opt.inner_iter, opt.damping)
    if opt.adapt_damping:
        name += '-ad'

root_dir = 'results/'+name
os.makedirs(root_dir, exist_ok=True)

# set global settings
def init_plotting():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.0 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.0 * plt.rcParams['font.size']

init_plotting()

# create logger
path = os.path.dirname(os.path.abspath(__file__))
path_file = os.path.join(path, 'follow_ridge_1D.py')
package_file = os.path.join(path, 'utils/misc.py')
logger = get_logger('log', logpath=root_dir+'/', filepath=path_file, package_files=[package_file])
logger.info(vars(opt))


def activation_fn(name):
    if name == 'elu':
        return tf.nn.elu
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.nn.tanh


def generator(x, output_dim=1, n_hidden=opt.g_hidden, n_layer=opt.g_layers,
              initializer=(tf.glorot_normal_initializer(seed=2020) if opt.init=='xavier'
              else tf.initializers.orthogonal(gain=1.0))):
    with tf.variable_scope("generator"):
        for i in range(n_layer):
            x = tf.layers.dense(x, n_hidden, activation=activation_fn(opt.g_act), kernel_initializer=initializer)
        x = tf.layers.dense(x, output_dim, activation=None, kernel_initializer=initializer)
    return x


def discriminator(x, n_hidden=opt.d_hidden, n_layer=opt.d_layers, reuse=False,
                  initializer=(tf.glorot_normal_initializer(seed=2020) if opt.init=='xavier'
                  else tf.initializers.orthogonal(gain=1.0))):
    with tf.variable_scope("discriminator", reuse=reuse):
        for i in range(n_layer):
            x = tf.layers.dense(x, n_hidden, activation=activation_fn(opt.d_act), kernel_initializer=initializer)
        x = tf.layers.dense(x, 1, activation=None, kernel_initializer=initializer)
    return x


rng = np.random.RandomState(seed=2020)
data_generator = MOG_1D(rng, std=opt.data_std)

tf.reset_default_graph()
tf.set_random_seed(2020)
sess = tf.Session()

real_samples = tf.placeholder(tf.float32, [None, opt.x_dim])
noise = tf.placeholder(tf.float32, [None, opt.z_dim])

# Construct generator and discriminator nets
fake_samples = generator(noise, output_dim=opt.x_dim)
real_score = discriminator(real_samples)
fake_score = discriminator(fake_samples, reuse=True)

# Standard GAN loss
loss_disc_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score,
                                                labels=tf.ones_like(real_score)))
loss_disc_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score,
                                                labels=tf.zeros_like(fake_score)))
loss_disc = loss_disc_real + loss_disc_fake
loss_gen = -loss_disc_fake # Saddle objective

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")

# define saver
saver = tf.train.Saver(var_list=gen_vars + disc_vars, max_to_keep=8)

# regularization
loss_disc += opt.weight_decay * tf.reduce_sum(flatten(disc_vars) ** 2)

# misc functions
gen_set_params = SetFromFlat(sess, gen_vars)
gen_get_params = GetFlat(sess, gen_vars)

disc_set_params = SetFromFlat(sess, disc_vars)
disc_get_params = GetFlat(sess, disc_vars)

disc_unflatten = unflatten(disc_vars)
gen_unflatten = unflatten(gen_vars)

# create optimizers
g_opt = RMSProp(opt.gen_learning_rate, gen_vars, name='g_rmsprop')
d_opt = RMSProp(opt.disc_learning_rate, disc_vars, name='d_rmsprop')

# gradient operator
gen_grads = tf.gradients(loss_gen, gen_vars)
gen_grads_flat = flatten(gen_grads)
disc_grads = tf.gradients(loss_disc, disc_vars)
disc_grads_flat = flatten(disc_grads)

# hessian-vector product
vecs_d = tf.placeholder(tf.float32, [None])
unflatten_vecs_d = disc_unflatten(vecs_d)
hvp = flatten(tf.gradients(disc_grads, disc_vars, grad_ys=unflatten_vecs_d))

d_num_params = flatten(disc_vars).get_shape().as_list()[0]
g_num_params = flatten(gen_vars).get_shape().as_list()[0]

# preconditioning
if opt.vanilla_gradient:
    gen_pred_grads_flat = gen_grads_flat
    disc_pred_grads_flat = disc_grads_flat
else:
    gen_pred_grads_flat = flatten(g_opt.preconditioning(list(zip(gen_grads, gen_vars))))
    disc_pred_grads_flat = flatten(g_opt.preconditioning(list(zip(disc_grads, disc_vars))))

# gradient norm
gradient_norm_g = tf.reduce_sum(gen_grads_flat ** 2)
gradient_norm_d = tf.reduce_sum(disc_grads_flat ** 2)


# training
sess.run(tf.global_variables_initializer())

# resume
if opt.resume:
    logger.info('==> Getting selection pattern from checkpoint..')
    latest_ckpt = tf.train.latest_checkpoint(opt.resume)
    logger.info(latest_ckpt)
    saver.restore(sess, latest_ckpt)

init_gen_params = gen_get_params()
init_disc_params = disc_get_params()


gnorm_list = []
dnorm_list = []

def sample_batch():
    z = rng.normal(size=[opt.batch_size, opt.z_dim])
    x = data_generator.sample(opt.batch_size)
    feed_dict = {real_samples: x, noise: z}
    return feed_dict


def single_update(itr, old_gen_velocity, old_disc_velocity, plot=False):
    global damping
    feed_dict = sample_batch()

    def hessian_vector_prod(p):
        feed_dict_ = {**feed_dict, **{vecs_d: p}}
        return sess.run(hvp, feed_dict=feed_dict_)

    g_grads = sess.run(gen_pred_grads_flat, feed_dict=feed_dict)

    old_gen_params = gen_get_params()
    gen_set_params(old_gen_params - opt.gen_learning_rate * g_grads)
    d_grads = sess.run(disc_pred_grads_flat, feed_dict=feed_dict)
    gen_set_params(old_gen_params)

    gen_update = gen_velocity = opt.gen_learning_rate * g_grads + opt.momentum * old_gen_velocity
    disc_update = disc_velocity = opt.disc_learning_rate * d_grads + opt.momentum * old_disc_velocity

    if opt.follow_ridge:
        # compute the correction term in FR
        old_gen_params = gen_get_params()
        old_disc_grads = sess.run(disc_grads_flat, feed_dict)
        gen_set_params(old_gen_params - gen_update)
        new_disc_grads = sess.run(disc_grads_flat, feed_dict)
        e_grads = (old_disc_grads - new_disc_grads) / opt.gen_learning_rate

        # take the Hessian inverse
        pred_e_grads = conjugate_gradient(hessian_vector_prod, e_grads, damping, max_iter=opt.inner_iter)
        pred_e_grads = np.nan_to_num(pred_e_grads)

        if opt.adapt_damping:
            reference = opt.gen_learning_rate ** 2 * np.sum(e_grads ** 2) + 1e-16
            q_model = (hessian_vector_prod(pred_e_grads) - e_grads) * opt.gen_learning_rate
            q_ratio = (reference - np.sum(q_model ** 2)) / reference

            old_disc_grads = sess.run(disc_grads_flat, feed_dict)
            old_disc_params = disc_get_params()
            disc_set_params(old_disc_params + opt.gen_learning_rate * pred_e_grads)
            new_disc_grads = sess.run(disc_grads_flat, feed_dict)
            disc_set_params(old_disc_params)
            true_model = new_disc_grads - old_disc_grads - opt.gen_learning_rate * e_grads
            true_ratio = (reference - np.sum(true_model ** 2)) / reference

            rel_ratio = true_ratio / (q_ratio + 1e-16)
            if itr % 100 == 0:
                logger.info('==> Iteration: %d' % itr)
                logger.info('==> Damping: %e' % damping)
                logger.info('\n')

            if opt.adapt_damping:
                if rel_ratio < 0.0 or true_ratio < 0.0:
                    pred_e_grads = 0.0 * pred_e_grads
                    damping = damping * 2
                elif rel_ratio < 0.5:
                    damping = damping * 1.1
                elif rel_ratio > 0.95:
                    damping = max(1e-8, damping * 0.9)
                damping = min(damping, opt.damping)

        # compute the gradient of disc at an extrapolated point
        old_disc_params = disc_get_params()
        disc_set_params(old_disc_params + opt.gen_learning_rate * pred_e_grads)
        d_grads = sess.run(disc_pred_grads_flat, feed_dict=feed_dict)
        disc_set_params(old_disc_params)
        disc_update = opt.disc_learning_rate * d_grads

        # reset old generator parameters
        gen_set_params(old_gen_params)

        disc_velocity = opt.momentum * old_disc_velocity + disc_update
        disc_update = disc_velocity - opt.gen_learning_rate * pred_e_grads

    if itr % 100 == 0 and plot:
        gnorm, dnorm = sess.run([gradient_norm_g, gradient_norm_d], feed_dict)
        gnorm_list.append(gnorm)
        dnorm_list.append(dnorm)

        plt.close()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(gnorm_list)
        ax1.set_yscale('log')
        ax2.plot(dnorm_list)
        ax2.set_yscale('log')
        plt.tight_layout()
        plt.savefig(root_dir+'/norm.pdf')

    return gen_update, disc_update, gen_velocity, disc_velocity

# training loop
damping = opt.damping
gen_velocity = disc_velocity = 0.0
for itr in tqdm(range(opt.iteration+1)):
    old_gen_params = gen_get_params()
    old_disc_params = disc_get_params()

    gen_update, disc_update, gen_velocity, disc_velocity = single_update(itr, gen_velocity, disc_velocity, plot=True)
    gen_set_params(old_gen_params - gen_update)
    disc_set_params(old_disc_params - disc_update)

    if itr % 5000 == 0:
        print("====> iteration: %d" % itr)
        samples = sess.run(fake_samples, feed_dict=sample_batch())
        plt.close()
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        sns.kdeplot(np.reshape(samples, [-1]), shade=True, ax=ax1)
        ax1.set_title('Generator Density')
        ax1.set_xlim([-6.0, 6.0])
        ax1.grid(linestyle='--')

        xx = np.linspace(-6, 6, 400)
        xz = sess.run(real_score, feed_dict={real_samples: np.reshape(xx, [-1, 1])})
        xz = 1. / (1. + np.exp(-xz))
        ax2.plot(xx, np.reshape(xz, [-1]), linewidth=2.5)
        ax2.set_title('Discriminator Prediction')
        ax2.set_ylim([0.0, 1.0])
        ax2.grid(linestyle='--')

        xx = np.linspace(0, itr, (itr//100) + 1)
        line1, = ax3.plot(xx, dnorm_list, linewidth=2.5)
        line2, = ax3.plot(xx, gnorm_list, linewidth=2.5)
        ax3.set_title('Gradient norm')
        ax3.set_yscale('log')
        ax3.legend((line1, line2), ('discriminator', 'generator'))
        ax3.grid(linestyle='--')
        plt.tight_layout()
        plt.savefig(root_dir+'/iter-%d.pdf' % itr)