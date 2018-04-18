"""
This file runs the linear regressor MVP code. The command line params 
can be seen in main().
"""

import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from model import FF
from gen_data import gen_data
from pathlib import Path
import os
import pickle
import json

NUM_INPUTS = 2
NUM_OUTPUTS = NUM_INPUTS

cwd = os.getcwd()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train')
  parser.add_argument('--num_inputs', default=NUM_INPUTS)
  parser.add_argument('--batch_size', default=3)
  parser.add_argument('--num_neurons', default=2)
  parser.add_argument('--num_layers', default=1)
  parser.add_argument('--num_outputs', default=NUM_OUTPUTS)
  parser.add_argument('--learning_rate', default=0.001)
  parser.add_argument('--num_epochs', default=10)
  parser.add_argument('--checkpoint_dir', default='./checkpoints')
  parser.add_argument('--restore_path', default=None)
  parser.add_argument('--vars_file', default=None)

  parser.add_argument('--ps_hosts', default='localhost:2222')
  parser.add_argument('--worker_hosts', default='localhost:2222')
  parser.add_argument('--job_name', default='worker')
  parser.add_argument('--task_index', default=0)

  args, _  = parser.parse_known_args()
  if args.mode == 'train':
    train(args)
  elif args.mode == 'test':
    test(args)

def restore_graph(sess,args):

  trained_model = tf.train.import_meta_graph(args.restore_path + '.meta')
  #trained_model.restore(sess, tf.train.latest_checkpoint(cwd + '/checkpoints/.'))
  trained_model.restore(sess, args.restore_path)

  w = []
  b = []
  model_vars_file = args.vars_file  # need to get back w's and b's for gen_data to reproduce same slopes and y-intercepts for lines
  with open(model_vars_file, 'rb') as f:
    w,b = pickle.load(f)
  print('\n\nModel and variables restored.\n\n')

  if (len(w) != int(args.num_inputs)):
    print('Error: num_inputs length %d does not equal stored variable length %d\n' % (int(args.num_inputs),len(w)))
    quit()

  return trained_model, w, b

def train(args):

  tf_config = None
  tf_config_json = None
  cluster = None
  job_name = None
  task_index = None
  ps_hosts = []
  worker_hosts = []
  config_file = False
  job_name = None
  task_index = 0

  try:
    print(os.environ['TF_CONFIG'])
    config_file = True
  except KeyError:
    pass

  if config_file:
    tf_config = os.environ.get('TF_CONFIG', '{}')
    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get('cluster', {})
    job_name = tf_config_json.get('task', {}).get('type', "")
    task_index = tf_config_json.get('task', {}).get('index', "")
    ps_hosts = cluster.get("ps")
    worker_hosts = cluster.get("worker")
  else:
    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    job_name = args.job_name
    task_index = args.task_index

  graph = tf.Graph()
  var_path = cwd + '/' + args.checkpoint_dir + '/variables/'

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=job_name,
                           task_index=task_index)

  if job_name == "ps":
    server.join()
  elif job_name == "worker":
  
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):

      with graph.as_default():

        # Graph object and scope created
        # ...now define all parts of the graph here
        feed_fwd_model = FF(args, graph)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Now that the graph is defined, create a session to begin running
        with tf.Session() as sess:

          sess.run(init)
          # Prepare to Save model
          i = 0
          model = 'model%s' % i
          ckpt_file_index = Path(cwd + '/' + args.checkpoint_dir + '/' + model + '.ckpt.index')
          ckpt_file = Path(cwd + '/' + args.checkpoint_dir + '/' + model + '.ckpt')
          while ckpt_file_index.is_file():
            i += 1
            model = 'model%s' % i
            ckpt_file_index = Path(cwd + '/' + args.checkpoint_dir + '/' + model + '.ckpt.index')
          ckpt_file = Path(cwd + '/' + args.checkpoint_dir + '/' + model + '.ckpt')

          num_epochs = int(args.num_epochs)
          y_acc = np.zeros((int(args.batch_size),int(args.num_outputs)))
          loss = None
          y_ = None
      
          w = []
          b = []
          if (args.restore_path != None):
            trained_model_saver, w, b = restore_graph(sess,args)
            print('...continuing training')

          # guards against accidental updates to the graph which can cause graph
          # increase and performance decay over time (with more iterations)
          sess.graph.finalize()

          for e in range(num_epochs):
            w, b, train_input, train_output = gen_data(int(args.batch_size),int(args.num_inputs), w, b)
            y_, loss, _ = sess.run(feed_fwd_model.run(), feed_dict={feed_fwd_model.x: train_input, feed_fwd_model.y: train_output})
            y_acc = y_
            threshold = 1000
            w_b_saved = False
            if ((e % 50) == 0):
              print('epoch: %d - loss: %2f' % (e,loss))
              if (e > 0 and (e % threshold == 0)):
                print('Writing checkpoint %d' % e)
                print(train_output, w, b)
                print('\n')
                print(y_acc, sess.run(feed_fwd_model.weights)[0], sess.run(feed_fwd_model.biases)[0])
                save_path = saver.save(sess, str(ckpt_file), global_step=e)
                if not w_b_saved:
                  try:
                    with open(var_path + model + '.pkl', 'wb') as f:
                      pickle.dump([w,b],f)
                      w_b_saved = True
                  except FileNotFoundError as fnf:
                    os.makedirs(var_path)
                    with open(var_path + model + '.pkl', 'wb') as f:
                      pickle.dump([w,b],f)
                      w_b_saved = True
          save_path = saver.save(sess, str(ckpt_file))
          if not w_b_saved:
            try:
              with open(var_path + model + '.pkl', 'wb') as f:
                pickle.dump([w,b],f)
            except FileNotFoundError as fnf:
              os.makedirs(var_path)
              with open(var_path + model + '.pkl', 'wb') as f:
                pickle.dump([w,b],f)
          print('Model saved to %s' % str(save_path))
          sess.close()

def test(args):

  inference_graph = tf.Graph()
  with tf.Session(graph=inference_graph) as sess:

    if not args.restore_path or not args.vars_file:
      print('\n\n\tSpecify a restore_path: --restore_path=<path_to_ckpt> and --vars_file=<vars_file_pathname>\n\n')
      quit()
    
    trained_model_saver, w, b = restore_graph(sess,args)

    _y_ = inference_graph.get_tensor_by_name('y_:0')
    _loss = inference_graph.get_tensor_by_name('loss:0')
    _x = inference_graph.get_tensor_by_name('x:0')
    _y = inference_graph.get_tensor_by_name('y:0')

    while(1):
      w, b, train_input, train_output = gen_data(int(args.batch_size),int(args.num_inputs), w, b)
      y_ = sess.run(_y_, feed_dict={_x: train_input, _y: train_output})
      loss = sess.run(_loss, feed_dict={_x: train_input, _y: train_output})
      y_acc = y_
      print('Mean Squared Error Loss: %2f\n' % loss)
      print(train_output)
      print('\n')
      print(y_acc)
      print('\n')
      input('Press Enter to continue...')

if __name__ == '__main__':
  #print(os.environ['TF_CONFIG'])
  main()
