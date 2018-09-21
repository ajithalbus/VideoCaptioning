import os
import sys
import numpy as np
import tensorflow as tf

# From lm_1b
import language_model.lm_1b.data_utils as data_utils

from six.moves       import xrange
from google.protobuf import text_format

#-------------------------------------------------------------------------------
# Adopted from lm_1b_eval.py
def LoadModel(gd_file, ckpt_file):
    """Load the model from GraphDef and Checkpoint.
    Args: gd_file: GraphDef proto text file. ckpt_file: TensorFlow Checkpoint file.
    Returns: TensorFlow session and tensors dict."""
    with tf.Graph().as_default():
        #class FastGFile: File I/O wrappers without thread locking.
        with tf.gfile.FastGFile(gd_file, 'r') as f:
            # Py 2: s = f.read().decode()
            s = f.read()
            # Serialized version of Graph
            gd = tf.GraphDef()
            # Merges an ASCII representation of a protocol message into a message.
            text_format.Merge(s, gd)

        tf.logging.info('Recovering Graph %s', gd_file)

        t = {}
        [t['states_init'], t['lstm/lstm_0/control_dependency'],
         t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
         t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
         t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
         t['all_embs'], t['softmax_weights'], t['global_step']
        ] = tf.import_graph_def(gd, {}, ['states_init',
                                         'lstm/lstm_0/control_dependency:0',
                                         'lstm/lstm_1/control_dependency:0',
                                         'softmax_out:0',
                                         'class_ids_out:0',
                                         'class_weights_out:0',
                                         'log_perplexity_out:0',
                                         'inputs_in:0',
                                         'targets_in:0',
                                         'target_weights_in:0',
                                         'char_inputs_in:0',
                                         'all_embs_out:0',
                                         'Reshape_3:0',
                                         'global_step:0'], name='')

	tgd=tf.import_graph_def(gd, {}, ['states_init',
                                         'lstm/lstm_0/control_dependency:0',
                                         'lstm/lstm_1/control_dependency:0',
                                         'softmax_out:0',
                                         'class_ids_out:0',
                                         'class_weights_out:0',
                                         'log_perplexity_out:0',
                                         'inputs_in:0',
                                         'targets_in:0',
                                         'target_weights_in:0',
                                         'char_inputs_in:0',
                                         'all_embs_out:0',
                                         'Reshape_3:0',
                                         'global_step:0'], name='')

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

    return sess, t,gd
