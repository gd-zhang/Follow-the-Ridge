import tensorflow as tf
import numpy as np

__all__ = ['flatten', 'unflatten', 'SetFromFlat', 'GetFlat', 'conjugate_gradient', 'gradient_descent']


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatten(tensors):
    if isinstance(tensors, (tuple, list)):
        return tf.concat(
            tuple(tf.reshape(tensor, [-1]) for tensor in tensors), axis=0)
    else:
        return tf.reshape(tensors, [-1])


class unflatten(object):
    def __init__(self, tensors_template):
        self.tensors_template = tensors_template

    def __call__(self, colvec):
        if isinstance(self.tensors_template, (tuple, list)):
            offset = 0
            tensors = []
            for tensor_template in self.tensors_template:
                sz = np.prod(tensor_template.shape.as_list(), dtype=np.int32)
                tensor = tf.reshape(colvec[offset:(offset + sz)],
                                           tensor_template.shape)
                tensors.append(tensor)
                offset += sz

            tensors = list(tensors)
        else:
            tensors = tf.reshape(colvec, self.tensors_template.shape)

        return tensors


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        shapes = map(var_shape, var_list)
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.op.eval(session=self.session)


def conjugate_gradient(hvp, b, damping=0.0, max_iter=30, tol=1e-8):
    x = np.zeros_like(b)

    # two hvp to optimize square error objective (instead of quadratic one)
    r = hvp(hvp(x)) + damping * x - hvp(b)
    p = - r
    init_rdotr = rdotr = r.dot(r)
    if init_rdotr == 0.0:
        return x
    for _ in range(max_iter):
        Ap = hvp(hvp(p)) + damping * p
        pAp = p.dot(Ap)
        v = rdotr / pAp
        x += v * p
        r += v * Ap
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = - r + mu * p
        rdotr = newrdotr

        if rdotr <= tol * init_rdotr:
            break

    return x


def gradient_descent(hvp, b, damping=0.0, max_iter=30):
    x = np.zeros_like(b)

    r = hvp(hvp(x)) + damping * x - hvp(b)
    p = - r
    for _ in range(max_iter):
        Ap = hvp(hvp(p)) + damping * p
        pAp = p.dot(Ap)
        # if pAp < 1e-6:
        #     break
        v = r.dot(p) / pAp
        x -= v * p
        r -= v * Ap
        p = - r

    return x
