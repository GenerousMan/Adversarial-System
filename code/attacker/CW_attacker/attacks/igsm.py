import tensorflow as tf
import numpy as np

def igsm(model, x, y, epochs=1.0, eps=1.0, clip_min=0.0, clip_max=1.0, min_proba=0.0, parallel = False, target=False):
    """
    :param model: A wrapper that returns the output tensor of the model.
    :param x: The input placeholder a 2D or 4D tensor.
    :param y: The desired class label for each input, either an integer or a
              list of integers.
    :param epochs: Maximum epochs to run.  When it is a floating number in [0,
        1], it is treated as the distortion factor, i.e., gamma in the original
        paper.
    :param eps: The noise added to input per epoch.
    :param clip_min: The minimum value in output tensor.
    :param clip_max: The maximum value in output tensor.
    :param min_proba: The minimum probability the model produces the desired
        target label given the adversarial samples.  The larger, the stronger
        the adversarial samples.  If this is set to >1.0, then add noise until
        the maximum epoch is reached.

    :return: A tensor, contains adversarial samples for each input.
    """
    xshape = tf.shape(x)
    n = xshape[0]
    target = tf.cond(tf.equal(1, tf.rank(y)),
                     lambda: tf.one_hot(y, 1000,dtype = tf.float32),
                     lambda: tf.cast(y,tf.float32))

    def _fn(i):
        # `xi` is of the shape (1, ....), the first dimension is the number of
        # samples, 1 in this case.  `yi` is just a scalar, denoting the target
        # class index.
        xi = tf.gather(x, [i])
        yi = tf.gather(target, [i])

        # `xadv` is of the shape (1, ...), same as xi.
        xadv = igsm_fn(model, xi, yi, eps=eps, epochs=epochs,
                        clip_min=clip_min, clip_max=clip_max,
                        min_proba=min_proba)
        return xadv[0]
    if parallel is False:
        return tf.map_fn(_fn, tf.range(n), dtype=tf.float32,name='fgsm_batch')
    else:
        return igsm_fn(model, x, target, eps=eps, epochs=epochs,
                        clip_min=clip_min, clip_max=clip_max,
                        min_proba=min_proba)
    


def igsm_fn(model, x, y, eps=0.01, epochs=1, clip_min=0., clip_max=1.,min_proba = 2.0):
    """
    Target class gradient sign method.

    See https://arxiv.org/pdf/1607.02533.pdf.  This method is similar to FGSM.
    The only difference is that

        1. TGSM allows to specify the desired label, i.e., targeted attack.

        2. Modified towards the least-likely class label when desired label is
           not specified.

    :param model: A model that returns the output as well as logits.
    :param x: The input placeholder.
    :param y: The desired target label, set to the least-likely class if not
              specified.
    :param eps: The noise scale factor.
    :param epochs: Maximum epoch to run.
    :param clip_min: Minimum value in output.
    :param clip_max: Maximum value in output.
    """
    x_adv = tf.identity(x)
    eps = np.abs(eps)
    #ybar, logits = model(x_adv, logits=True)
    

    def _cond(x_adv, i):
        ybar = model(x_adv)
        ybar = tf.squeeze(ybar)
        proba = tf.reduce_mean(ybar*y)

        return tf.reduce_all([tf.less(i, epochs),
                              tf.less(proba, min_proba)])

    def _body(x_adv, i):
        _, logits = model(x_adv, logits=True)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        dy_dx, = tf.gradients(loss, x_adv)
        x_adv = tf.stop_gradient(x_adv + eps*tf.sign(dy_dx))
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        return x_adv, i+1

    x_adv, _ = tf.while_loop(_cond, _body, (x_adv, 0), back_prop=False,
                             name='igsm')
    return x_adv
