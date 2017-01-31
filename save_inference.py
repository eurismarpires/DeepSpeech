import tensorflow as tf

from tensorflow.python.ops import ctc_ops
from DeepSpeech import *
from os import getcwd
from subprocess import call

BASEDIR = getcwd()
FREEZE = True
QUANTIZE = False

# Input tensor will be of shape [batch_size, n_steps, n_input + 2*n_input*n_context]
input_tensor = tf.placeholder(tf.float32, [1, None, n_input + 2*n_input*n_context], name="x")
seq_length = tf.placeholder(tf.int32, [1], name="x_length")

# Calculate the logits of the batch using BiRNN
logits = BiRNN(input_tensor, seq_length, 0)

# Beam search decode the batch
decoded, _ = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)
decoded = tf.convert_to_tensor(
    [tf.sparse_tensor_to_dense(sparse_tensor) for sparse_tensor in decoded])

decoded = tf.identity(decoded, name="y")

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    with open("graph-def/inference-graph.pb", "wb") as fout:
        fout.write(sess.graph.as_graph_def().SerializeToString())

    print("Written graph def file.")
    
    state = tf.train.get_checkpoint_state(checkpoint_dir)
    if not state or not state.model_checkpoint_path:
        print("No checkpoint found.")
        exit(0)

    if FREEZE:
        call(["python", "../tensorflow/tensorflow/python/tools/freeze_graph.py",
              "--input_graph=graph-def/inference-graph.pb",
              "--input_binary",
              "--input_checkpoint={}".format(state.model_checkpoint_path),
              "--output_graph=graph-def/frozen-graph.pb",
              "--output_node_names=y"])
        print("Wrote frozen graph to graph-def/frozen-graph.pb")
    
    if QUANTIZE:
        call(["python", "../tensorflow/tensorflow/tools/quantization/quantize_graph.py",
              "--input=graph-def/{}-graph.pb".format("frozen" if FREEZE else "inference"),
              "--output_node_names=y",
              "--output=graph-def/quantized-graph.pb",
              "--mode=eightbit"])
        print("Wrote quantized graph to graph-def/quantized-graph.pb")
