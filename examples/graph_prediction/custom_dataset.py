"""
This example shows how to define your own dataset and use it to train a
non-trivial GNN with message-passing and pooling layers.
The script also shows how to implement fast training and evaluation functions
in disjoint mode, with early stopping and accuracy monitoring.

The dataset that we create is a simple synthetic task in which we have random
graphs with randomly-colored nodes. The goal is to classify each graph with the
color that occurs the most on its nodes. For example, given a graph with 2
colors and 3 nodes:

x = [[1, 0],
     [1, 0],
     [0, 1]],

the corresponding target will be [1, 0].
"""

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

from spektral.models import GeneralGNN
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj

################################################################################
# Config
################################################################################
learning_rate = 1e-2  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 10  # Patience for early stopping
batch_size = 32  # Batch size

def plotSKlearnGraph(cg, df, dotsize=5):
    """
    given a graph cg created from a sklearn function, plot its nodes and edges
    @param cg:
    @return:
    """
    A = cg.toarray()
    edge_start, edge_end = np.where(A==1)
    for s, e in zip(edge_start, edge_end):
        plt.plot([df.iloc[s]['X'], df.iloc[e]['X']], [df.iloc[s]['Y'], df.iloc[e]['Y']], 'k-', zorder=1)
    plt.scatter(df['X'], df['Y'], s=dotsize, zorder=2)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
    return

################################################################################
# Load data
################################################################################
class MyDataset(Dataset):
    """
    A dataset of random colored graphs.
    The task is to classify each graph with the color which occurs the most in
    its nodes.
    The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
    nodes connected with probability `p`.
    """

    def __init__(self, n_samples, n_colors=3, n_min=10, n_max=100, p=0.1, **kwargs):
        self.n_samples = n_samples
        self.n_colors = n_colors
        self.n_min = n_min
        self.n_max = n_max
        self.p = p
        super().__init__(**kwargs)

    def read(self):
        def make_graph():
            n = np.random.randint(self.n_min, self.n_max)
            colors = np.random.randint(0, self.n_colors, size=n)

            # Node features
            x = np.zeros((n, self.n_colors))
            x[np.arange(n), colors] = 1

            # Edges
            a = np.random.rand(n, n) <= self.p
            a = np.maximum(a, a.T).astype(int)
            a = sp.csr_matrix(a)

            # Labels
            y = np.zeros((self.n_colors,))
            color_counts = x.sum(0)
            y[np.argmax(color_counts)] = 1

            return Graph(x=x, a=a, y=y)

        # We must return a list of Graph objects
        return [make_graph() for _ in range(self.n_samples)]

class MyDataset2(Dataset):
    """
    A dataset of KNN-graphs with multiple features per node (i.e. color values).
    The task is to classify each graph with the color which occurs the most in
    its nodes.
    The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
    nodes connected with probability `p`.
    """

    def __init__(self, n_samples, n_colors=3, n_min=10, n_max=100, knn=5, max_edge_length=None, **kwargs):
        self.n_samples = n_samples  # how many graphs to create
        self.n_colors = n_colors  # how many "color" features each node should have
        self.n_min = n_min  # minimum number of nodes of each graph
        self.n_max = n_max  # maximum number of nodes of each graph
        self.knn = knn  # how many edges each node should have
        self.max_edge_length = max_edge_length  # maximum allowed edge lengths
        super().__init__(**kwargs)

    def read(self):
        def make_graph(graphnumber):
            print("Loading data for graph nr. {}".format(graphnumber))
            n = np.random.randint(self.n_min, self.n_max)  # number of nodes

            # Node features discrete
            # colors = np.random.randint(0, self.n_colors, size=n)  # random color index for each node
            # x = np.zeros((n, self.n_colors))
            # x[np.arange(n), colors] = 1

            # Node features continuous
            x = np.random.rand(n, self.n_colors)  # each node gets random color values for each available color
            # colors = np.argmax(x, axis=1)  # only for fake graph labels, picks the index of the dominant color for each node

            # create random XY positions for the nodes
            nodepos = pd.DataFrame({'X': np.random.rand(n), 'Y': np.random.rand(n)})  # X, Y

            # Edges (without weights) represented as adjacency matrix
            a = kneighbors_graph(nodepos, self.knn)
            # n_edges_orig = a.count_nonzero()
            # print("Original knn edges: {}".format(n_edges_orig))

            # get edge lengths and prune edges that are too long
            n_edge_features = 1  # we only use the length of the edge as edge feature
            e = np.zeros((n, n, n_edge_features), dtype=float)
            n_pruned_edges = 0
            for (i, j) in zip(a.nonzero()[0], a.nonzero()[1]):
                n1 = nodepos.iloc[i]
                n2 = nodepos.iloc[j]
                elength = ((n1['X'] - n2['X'])**2 + (n1['Y'] - n2['Y'])**2) ** 0.5
                if self.max_edge_length is None or elength <= self.max_edge_length:
                    e[i, j, 0] = elength
                else:
                    n_pruned_edges += 1
                    e[i, j, 0] = 0.
                    a[i, j] = 0.
            # print("pruned eges: {}".format(n_pruned_edges))
            # n_edges = a.count_nonzero()
            # print("remaining edges: {}".format(n_edges))
            # plotSKlearnGraph(a, nodepos, dotsize=3)
            a = sp.csr_matrix(a)

            # Labels, count the number of maximum sum of color values
            y = np.zeros((self.n_colors,))
            color_counts = x.sum(0)
            y[np.argmax(color_counts)] = 1

            return Graph(x=x, a=a, e=None, y=y)

        # We must return a list of Graph objects
        return [make_graph(i) for i in range(self.n_samples)]


tmpGraphDataFile = "graphdata.tmp"
if os.path.exists(tmpGraphDataFile):
    print("Loading graph data set from file ...")
    data = pickle.load(open(tmpGraphDataFile, "rb"))
else:
    print("Creating graph data set ...")
    # delete the tempoprary graph data file if you want to load new data
    data = MyDataset2(2000, n_min=20, n_max=30, transforms=NormalizeAdj(), n_colors=6, knn=5, max_edge_length=0.4)
    pickle.dump(data, open(tmpGraphDataFile, "wb"))

# Train/valid/test split
idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)


################################################################################
# Build model
################################################################################
class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCSConv(32, activation="relu")
        self.conv2 = GCSConv(32, activation="relu")
        self.conv3 = GCSConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(data.n_labels, activation="softmax")

    def call(self, inputs):
        # use this if you have not defined any edge weights
        x, a, i = inputs  # node features, adjacency matrix, classes

        # use this if your graph also has some edge weights
        # todo I don't know yet how to use the edge weights, but its fine to train without them!
        # x, a, e, i = inputs  # node features, adjacency matrix, edge weights, classes

        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        output = self.global_pool([x, i])
        output = self.dense(output)

        return output


# SMALL MODEL
# model = Net()
# optimizer = Adam(lr=learning_rate)
# loss_fn = CategoricalCrossentropy()

# GENERAL MODEL
model = GeneralGNN(data.n_labels, activation="softmax")
optimizer = Adam(learning_rate)
loss_fn = CategoricalCrossentropy()

################################################################################
# Fit model
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])


epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, val_acc
            )
        )

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

################################################################################
# Evaluate model
################################################################################
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))
