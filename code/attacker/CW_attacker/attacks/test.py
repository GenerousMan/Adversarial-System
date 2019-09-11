from fgsm import fgsm
import tensorflow as tf 
import tensorflow.contrib.slim as slim
def model(x,logits):
    logits_ = slim.fullyconnected(x,10)
    softmax_ = tf.nn.softmax(logits_)
    if logits:
        return softmax_,logits_
    return softmax_

if __name__ == '__main__':
    x = tf.placeholder((None,28,28,1),dtype = tf.float32)
    y = tf.placeholder(1)

    output = model(x)

    img_adv = fgsm(model,x)

    