import tensorflow as tf

from DeepSpeech import *
from os import path, getcwd
from tensorflow.python.ops import ctc_ops
from util.audio import audiofile_to_input_vector
from util.text import ndarray_to_text

BASEDIR = getcwd()
input_graphdef = "graph-def/frozen-graph.pb"

graph = tf.Graph()
with graph.as_default():
    pb = tf.GraphDef()
    with open(input_graphdef, "rb") as fin:
        pb.ParseFromString(fin.read())

    imports = tf.import_graph_def(pb, name="")

    x = graph.get_tensor_by_name("x:0")
    seq_length = graph.get_tensor_by_name("x_length:0")
    y = graph.get_tensor_by_name("y:0")

    audio_in = audiofile_to_input_vector("data/ldc93s1/LDC93S1.wav", n_input, n_context)

    sess = tf.Session()

    # If restoring from an unfrozen graph, we also need to restore weights
    # vars_ = [graph.get_tensor_by_name(name + ":0") for name in ["h1", "b1", "h2", "b2", "h3", "b3", "h5", "b5", "h6", "b6", "BiRNN/FW/BasicLSTMCell/Linear/Matrix", "BiRNN/FW/BasicLSTMCell/Linear/Bias", "BiRNN/BW/BasicLSTMCell/Linear/Matrix", "BiRNN/BW/BasicLSTMCell/Linear/Bias"]]
    # saver = tf.train.Saver(vars_)
    # saver.restore(sess, "simpler-checkpoint/model.ckpt-1")

    decoded = sess.run([y], feed_dict={x: [audio_in], seq_length: [len(audio_in)]})
    print("text: {}".format(ndarray_to_text(decoded[0][0][0])))

