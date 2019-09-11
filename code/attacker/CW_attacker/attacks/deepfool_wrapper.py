import tensorflow as tf
import numpy as np
from keras.models import Model
from attacks.deepfool import deepfool

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

def override_params(default, update):
    for key in default:
        if key in update:
            val = update[key]
            default[key] = val
            del update[key]

    if len(update) > 0:
        warnings.warn("Ignored arguments: %s" % update.keys())
    return default


# def prepare_attack(sess, model, x, Y):
#     nb_classes = 1000
#
#     f = model.predict
#
#     # model_logits = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
#
#     persisted_input = x
#     persisted_output = f(x)
#
#     print('>> Compiling the gradient tensorflow functions. This might take some time...')
#     scalar_out = [tf.slice(persisted_output, [0, i], [1, 1]) for i in range(0, nb_classes)]
#     print(tf.shape(scalar_out))
#     print('>> scalar_out finished')
#     dydx = [tf.gradients(scalar_out[i], [persisted_input])[0] for i in range(0, nb_classes)]
#
#     print('>> Computing gradient function...')
#     def grad_fs(image_inp, inds): return [sess.run(dydx[i], feed_dict={persisted_input: image_inp}) for i in inds]
#
#     return f, grad_fs



def generate_deepfool_examples(sess, model, X):
    """
    Untargeted attack. Y is not needed.
    """
    params = {'num_classes': 10, 'overshoot': 0.02, 'max_iter': 50}

    adv_x_list = []

    # Loop over the samples we want to perturb into adversarial examples
    for i in range(X.shape[0]):
        image = X[i:i+1,:,:,:]#i:i+1
        pert_image = deepfool(sess, image, model, **params)
        adv_x_list.append(pert_image)
    return np.vstack(adv_x_list)



# def generate_universal_perturbation_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
#     """
#     Untargeted attack. Y is not needed.
#     """
#
#     # TODO: insert a uint8 filter to f.
#     f, grad_fs = prepare_attack(sess, model, x, y, X, Y)
#
#     params = {'delta': 0.2,
#               'max_iter_uni': np.inf,
#               'xi': 10,
#               'p': np.inf,
#               'num_classes': 10,
#               'overshoot': 0.02,
#               'max_iter_df': 10,
#               }
#
#     params = override_params(params, attack_params)
#
#     # if not verbose:
#     #     # disablePrint(attack_log_fpath)
#
#     # X is randomly shuffled in unipert.
#     X_copy = X.copy()
#     v = universal_perturbation(X_copy, f, grad_fs, **params)
#     del X_copy
#
#     # if not verbose:
#     #     # enablePrint()
#
#     return X + v
