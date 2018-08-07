import json
import random

import numpy as np
import tensorflow as tf

from train import dropout, embed, gelu, swish, mask_attn_weights, split_heads, merge_heads, conv1d, norm
from text_utils import TextEncoder
from utils import find_trainable_variables, shape_list

act_fns = {
    'relu': tf.nn.relu,
    'swish': swish,
    'gelu': gelu
}

SEED = 42
N_CTX = 128  # 512
N_EMBD = 768
N_HEAD = 12
N_LAYER = 12
EMBD_PDROP = 0.1
ATTN_PDROP = 0.1
RESID_PDROP = 0.1
AFN = 'gelu'
ENCODER_PATH = 'model/encoder_bpe_40000.json'
BPE_PATH = 'model/vocab_40000.bpe'
N_TRANSFER = 12


random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

TEXT_ENCODER = TextEncoder(ENCODER_PATH, BPE_PATH)
ENCODER = TEXT_ENCODER.encoder
N_VOCAB = len(TEXT_ENCODER.encoder)

# parser.add_argument('--n_batch', type=int, default=8)
# parser.add_argument('--n_gpu', type=int, default=4)
# parser.add_argument('--lm_coef', type=float, default=0.5)


def transform_texts(list_of_texts):
    tokens = TEXT_ENCODER.encode(list_of_texts, verbose=False)
    n_batch = len(tokens)
    xmb = np.zeros((n_batch, N_CTX, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, N_CTX), dtype=np.float32)
    for i, x in enumerate(tokens):
        x1 = x[:N_CTX]
        l1 = len(x1)
        print(f"length: {l1}")
        xmb[i, :l1, 0] = x1
        mmb[i, :l1] = 1
    xmb[:, :, 1] = np.arange(N_VOCAB, N_VOCAB+N_CTX)
    return xmb, mmb


def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, ATTN_PDROP, train)

    a = tf.matmul(w, v)
    return a


def attn(x, scope, n_state, train=False, scale=False):
    assert n_state % N_HEAD == 0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, N_HEAD)
        k = split_heads(k, N_HEAD, k=True)
        v = split_heads(v, N_HEAD)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train)
        a = dropout(a, RESID_PDROP, train)
        return a


def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[AFN]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train)
        h2 = dropout(h2, RESID_PDROP, train)
        return h2


def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, train=train, scale=scale)
        n = norm(x+a, 'ln_1')
        m = mlp(n, 'mlp', nx*4, train=train)
        h = norm(n+m, 'ln_2')
        return h


def model(X, M, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [N_VOCAB + N_CTX, N_EMBD],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, EMBD_PDROP, train)

        X = tf.reshape(X, [-1, N_CTX, 2])
        M = tf.reshape(M, [-1, N_CTX])

        h = embed(X, we)
        for layer in range(N_LAYER):
            h = block(h, 'h%d' % layer, train=train, scale=True)

        lm_h = tf.reshape(h, [-1, N_EMBD])
        lm_logits = tf.reshape(
            tf.matmul(lm_h, we[:N_VOCAB, :], transpose_b=True),
            [-1, N_CTX, N_VOCAB]
        )
        lm_logits_truncated = tf.reshape(
            lm_logits[:, :-1],
            [-1, N_VOCAB]
        )
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=lm_logits_truncated, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(
            lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
        lm_losses = tf.reduce_sum(
            lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
        return lm_logits, lm_losses


def build_graph(sess):
    X = tf.placeholder(tf.int32, [None, N_CTX, 2])
    M = tf.placeholder(tf.float32, [None, N_CTX])
    lm_logits, lm_losses = model(X, M, train=False, reuse=False)
    params = find_trainable_variables('model')
    sess.run(tf.global_variables_initializer())
    shapes = json.load(open('model/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load('model/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape)
                   for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:N_CTX]
    init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
    del init_params[1]
    n_transfer = 1 + N_TRANSFER * 12
    sess.run([p.assign(ip)
              for p, ip in zip(
        params[:n_transfer],
        init_params[:n_transfer])])
    return X, M, lm_logits, lm_losses
