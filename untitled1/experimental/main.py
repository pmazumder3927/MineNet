import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import untitled1.experimental.util as util
import untitled1.experimental.nbtvae3d as vae3d
import untitled1.StructureManager as sm

import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
# import skimage.measure as sk

from mpl_toolkits import mplot3d
scalar = 1
epochs = 500;
dataset_list = sm.load_structure_blocks('C:\\Users\\Danny\\Documents\\structures32', [32, 32, 32])
X = np.subtract(np.multiply(np.divide(dataset_list, scalar),2),1)
X = np.expand_dims(X.astype(float), 4)
bf = util.BatchFeeder(X, 32)
util.plotVoxel(bf.next()[0], size=(3, 3))


model = vae3d.VAE3D(latent_dim=50)
model.train(bf, epochs + 1)


def generate_and_save_structures(model, epoch, test_input):
    predictions = model(test_input, training=False)
    processed_predictions = np.divide(np.multiply(np.add(predictions, 1), scalar), 2)
    processed_predictions = processed_predictions.astype(int)
    processed_predictions = processed_predictions[1,:,:,:]
    print(len(processed_predictions[processed_predictions < 0]))
    sm.create_nbt_from_3d(processed_predictions, epoch)



kld = []
rec = []
for e in range(len(model.learning_curve)):
    kld.append(np.mean(model.learning_curve[e]["kld"]))
    rec.append(np.mean(model.learning_curve[e]["rec"]))

plt.figure(figsize=(8, 2))

plt.subplot(1, 2, 1)
plt.plot(kld)
plt.title("KL-divergence")
plt.xlabel("epochs")
# plt.yscale("log")

plt.subplot(1, 2, 2)
plt.plot(rec)
plt.title("Reconstruction Error")
plt.xlabel("epochs")
# plt.yscale("log")

plt.show()


def getplotable(d, th=0.5, size=(6, 6)):
    temp = []
    bina = d > th
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            for k in range(d.shape[2]):
                if bina[i, j, k]:
                    temp.append([i, j, k])
    temp = np.array(temp)
    return temp


'''
index1 = np.random.randint(10000)
x1 = model.record["reconstructed"][index1]
z1 = model.record["z"][index1]

index2 = np.random.randint(10000)
x2 = model.record["reconstructed"][index2]
z2 = model.record["z"][index2]


print
index1, index2


vecs = util.interp(z1, z2, 9)

fig = plt.figure(figsize=(10, 4))
for i in range(len(vecs)):
    vec = vecs[i]
    _in = np.zeros((32, 200))
    _in[0, :] = vec
    _out = model.decode(_in)

    temp = getplotable(_out[0])

    ax = plt.subplot(2, 5, i + 1, projection='3d')

    n = 100
    colors = sns.color_palette("hls", 5)
    ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], c=colors[0], marker=".", alpha=0.5, linewidth=0, s=100)

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
plt.savefig("figure/vae3d/" + str(index1) + "_" + str(index2) + ".png")
plt.show()
'''
