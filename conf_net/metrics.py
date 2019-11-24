import tensorflow as tf

def softmax_correlation(pred, label, conf, thres=0.6, name='softmax_correlation'):
    with tf.name_scope(name):
        pred = tf.argmax(pred, axis=-1)
        label = tf.argmax(label, axis=-1)

        is_correct = tf.cast(tf.expand_dims(tf.equal(pred, label), axis=-1), dtype='float32')
        is_wrong   = tf.abs(is_correct - 1)

        is_high_conf = tf.cast(tf.greater(conf, thres), dtype='float32')
        is_low_conf   = tf.abs(is_high_conf - 1)

        
        f1_score, precision, recall = complete_f1_score(is_low_conf, is_wrong)

    return f1_score, precision, recall


def complete_f1_score(pred, label, name='f1_score'):
    with tf.name_scope(name):
        true_positive = tf.count_nonzero(pred * label)
        # true_negative  = tf.count_nonzero((pred-1) * (label-1))
        false_positive = tf.count_nonzero(pred * (label-1))
        false_negative = tf.count_nonzero((pred-1) * label)

        precision = true_positive / (true_positive + false_positive)
        recall    = true_positive / (true_positive + false_negative)

        f1_score = 2 * precision * recall / (precision + recall)

    return f1_score, precision, recall