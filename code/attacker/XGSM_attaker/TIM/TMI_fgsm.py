
import warnings

import numpy as np
import tensorflow as tf

from fast_gradient_method import optimize_linear
from compat import reduce_sum, reduce_mean, softmax_cross_entropy_with_logits
import utils_tf


def Tmi_fgsm(model,x,y,y_target=False,decay_factor=1.0,
			nb_iter=5,eps=0.5,eps_iter=0.1,ord=np.inf,
			clip_min=None,clip_max=None,sanity_checks=True):
    # Initialize loop variables
    momentum = tf.zeros_like(x)
    adv_x = x
    # Fix labels to the first model predictions for loss computation
    
#===================probs,not logits==============
    # y= model.predict(x,False)
#=================================================

    # y = y / reduce_sum(y, 1, keepdims=True)
    targeted = y_target

    def cond(i, _, __):
      return tf.less(i, nb_iter)

    def body(i, ax, m):
      logits = model.predict(ax,True)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      # Define gradient of loss wrt input
      grad, = tf.gradients(loss, ax)

      # Normalize current gradient and add it to the accumulated gradient
      red_ind = list(range(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-12, grad.dtype)
      grad = grad / tf.maximum(
          avoid_zero_div,
          reduce_mean(tf.abs(grad), red_ind, keepdims=True))
      m = decay_factor * m + grad

      optimal_perturbation = optimize_linear(m, eps_iter, ord)
      if ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      # Update and clip adversarial example in current iteration
      ax = ax + optimal_perturbation
      ax = x + utils_tf.clip_eta(ax - x, ord, eps)

      if clip_min is not None and clip_max is not None:
        ax = utils_tf.clip_by_value(ax, clip_min, clip_max)

      ax = tf.stop_gradient(ax)

      return i + 1, ax, m

    _, adv_x, _ = tf.while_loop(
        cond, body, (tf.zeros([]), adv_x, momentum), back_prop=True,
        maximum_iterations=nb_iter)

    return adv_x