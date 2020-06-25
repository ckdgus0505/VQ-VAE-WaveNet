from model import VQVAE
from Encoder.encoder import *
from Decoder.decoder import *
from dataset import *
from testset import *
from utils import display_time, suppress_tf_warning
import tensorflow as tf
import numpy as np
import time, os, sys, json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

suppress_tf_warning()

parser = ArgumentParser()
parser.add_argument('-dataset', default='VCTK', type=str,
                    help='VCTK or LibriSpeech or Aishell',
                    metavar='DATASET')
parser.add_argument('-length', default=6656, type=int,
                    dest='max_len', metavar='int',
                    help='number of samples one audio will contain')
parser.add_argument('-epoch', default=23, type=int,
                    dest='num_steps', metavar='int',
                    help='number of steps to train')
parser.add_argument('-batch', default=8, type=int,
                    dest='batch_size', metavar='int',
                    help='batch size')
parser.add_argument('-interval', default=200, type=int, 
                    dest='interval', metavar='int',
                    help='save log every interval step')
parser.add_argument('-restore',
                    dest='restore_path', metavar='string',
                    help='path to restore weights')
parser.add_argument('-save', default='saved_model/weights', 
                    dest='save_path', metavar='string',
                    help='path to save weights')
parser.add_argument('-params', default='model_parameters.json',
                    dest='parameter_path', metavar='str',
                    help='path to parameters file')
args = parser.parse_args()

dataset_args = {
    'relative_path': '../data/',
    'batch_size': args.batch_size,
    'max_len': args.max_len
}

if args.dataset == 'VCTK':
    dataset = VCTK(**dataset_args)
    testset = VCTK_test(**dataset_args)
else:
    raise NotImplementedError('dataset %s not implemented' % args.dataset)

with open(args.parameter_path) as file:
    parameters = json.load(file)
encoders = {'Magenta': Encoder_Magenta, '64': Encoder_64, '2019': Encoder_2019}
if parameters['encoder'] in encoders:
    encoder = encoders[parameters['encoder']](parameters['latent_dim'])
else:
    raise NotImplementedError('encoder %s not implemented' % args.encoder)
decoder = WavenetDecoder(parameters['wavenet_parameters'])
model_args = {
    'x': dataset.x,
    'speaker': dataset.y,
    'test_x': testset.x,
    'test_speaker': testset.y,
    'encoder': encoder,
    'decoder': decoder,
    'k': parameters['k'],
    'beta': parameters['beta'],
    'verbose': parameters['verbose'],
    'use_vq': parameters['use_vq'],
    'speaker_embedding': parameters['speaker_embedding'],
    'num_speakers': dataset.num_speakers,
    'dataset_size': dataset.total,
    'testset_size': testset.total
}

schedule = {int(k): v for k, v in parameters['learning_rate_schedule'].items()}

model = VQVAE(model_args)
model.build(learning_rate_schedule=schedule)

sess = tf.Session()
saver = tf.train.Saver()

if args.restore_path is not None:
    saver.restore(sess, args.restore_path)
else:
    sess.run(tf.global_variables_initializer())

gs = sess.run(model.global_step)
lr = sess.run(model.lr)
print('[restore] last global step: %d, learning rate: %.5f' % (gs, lr))

save_path = args.save_path
save_dir, save_name = save_path.split('/')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

writer = tf.summary.FileWriter(save_dir, sess.graph)

sess.run(dataset.init)
sess.run(testset.init)
xaxis = []
laxis = []
rclaxis = []
cmlaxis = []
vqlaxis = []
tlaxis = []
trclaxis = []
tcmlaxis = []
tvqlaxis = []
plt.xlabel("epoch")
plt.ylabel("loss")
plt.figure(figsize=(15,20))
for epoch in range (1,args.num_steps):
    mp1 = 0.0
    mp2 = 0.0
    mp3 = 0.0
    mp4 = 0.0
    for step in range(1, int(dataset.total/args.batch_size)):
        try:
            t = time.time()
            _, rl, gs, lr, cml, vql, l = sess.run([model.train_op, 
                                               model.reconstruction_loss, 
                                               model.global_step, 
                                               model.lr,
                                               model.commitment_loss,
                                               model.vq_loss,
                                               model.loss])
            mp1 += l
            mp2 += rl
            mp3 += cml
            mp4 += vql
            t = time.time() - t
            progress = '\r[step %d] %.2f' % (gs, step / args.num_steps * 100) + '%'
            loss = ' [recons %.5f] [lr %.8f]' % (rl, lr)
            second = (args.num_steps - step) * t
            print(progress + loss + display_time(t, second), end='')
        except tf.errors.OutOfRangeError:
            print('\ndataset re-initialised')
            sess.run(dataset.init)
            sess.run(testset.init)
    t = time.time()
    laxis.append(mp1/int(dataset.total/args.batch_size))
    rclaxis.append(mp2/int(dataset.total/args.batch_size))
    cmlaxis.append(mp3/int(dataset.total/args.batch_size))
    vqlaxis.append(mp4/int(dataset.total/args.batch_size))

    tl = 0.0
    trcl = 0.0
    tcml = 0.0
    tvql = 0.0
    tmp1 = 0.0
    tmp2 = 0.0
    tmp3 = 0.0
    tmp4 = 0.0
    for i in range (0, int(testset.total/args.batch_size)):
        tmp1, tmp2, tmp3, tmp4 = sess.run([model.test_loss, model.test_reconstruction_loss, model.test_commitment_loss, model.test_vq_loss])
        tl += tmp1
        trcl += tmp2
        tcml += tmp3
        tvql += tmp4
    
    tlaxis.append(tl/int(testset.total/args.batch_size))
    trclaxis.append(trcl/int(testset.total/args.batch_size))
    tcmlaxis.append(tcml/int(testset.total/args.batch_size))
    tvqlaxis.append(tvql/int(testset.total/args.batch_size))
    xaxis.append(epoch)
    plt.subplot(421)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, laxis)
    plt.subplot(422)
    plt.title("reconstruction_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, rclaxis)
    plt.subplot(423)
    plt.title("commitment_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, cmlaxis)
    plt.subplot(424)
    plt.title("vq_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, vqlaxis)

    plt.subplot(425)
    plt.title("test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, tlaxis)
    plt.subplot(426)
    plt.title("test_reconstruction_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, trclaxis)
    plt.subplot(427)
    plt.title("test_commitment_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, tcmlaxis)
    plt.subplot(428)
    plt.title("test_vq_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(xaxis, tvqlaxis)

    plt.savefig(save_dir+'/'+str(epoch)+'.png', dpi=400)
    summary = sess.run(model.summary)
    
    writer.add_summary(summary, epoch)
            
    t = time.time() - t
    progress = '\r[step %d] %.2f' % (gs, step / args.num_steps * 100) + '%'
    loss = ' [recons %.5f] [lr %.8f]' % (rl, lr)
    second = (args.num_steps * dataset.total - step) * t
    print(progress + loss + display_time(t, second), end='')
    
saver.save(sess, save_path, global_step=model.global_step)
