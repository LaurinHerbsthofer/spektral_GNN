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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from custom_data_load_functions import *
import pickle
import yaml
import datetime
import json

from spektral.models import GeneralGNN
from spektral.data import DisjointLoader

################################################################################
# Config
################################################################################

with open('settings.yaml') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outfolder = os.path.join(settings['outfolder'], timestamp)
    os.makedirs(outfolder)
    if settings['developmentrun']:
        always_load_data_anew = True  # [dev: True]
        load_n_rows = 10
        n_classes = 2
    else:
        always_load_data_anew = False
        load_n_rows = None
        n_classes = None


if os.path.exists(settings['tmpGraphDataFile_tr_va']) and os.path.exists(settings['tmpGraphDataFile_te']) and always_load_data_anew is False:
    print("Loading graph data set from file ...")
    try:
        data_tr_va = pickle.load(open(settings['tmpGraphDataFile_tr_va'], "rb"))
        print(" > Loaded {} training/validation samples.".format(len(data_tr_va)))
        data_te = pickle.load(open(settings['tmpGraphDataFile_te'], "rb"))
        print(" > Loaded {} test samples.".format(len(data_te)))
    except:
        assert False, "You might need to rebuild your graph data (delete the tmp file), possibly due to using a different python environment."
else:
    print("Creating graph data set ...")
    # delete the tempoprary graph data file if you want to load new data
    # data = MyDataset2(1300, n_min=2000, n_max=3000, transforms=NormalizeAdj(), n_colors=6, knn=5, max_edge_length=0.4)
    data_tr_va = cellGraphDataset(metadatafile=settings['metadata_train'],
                                  path_col=settings['path_col'],
                                  class_col=settings['class_col'],
                                  colList=settings['colList'],
                                  colNormalisation=settings['colNormalisation'],
                                  colListNewNames=settings['colListNewNames'],
                                  micronsPerPixel=settings['micronsPerPixel'],
                                  knn=settings['knn'],
                                  max_edge_length=settings['max_edge_length'],
                                  load_n_rows=load_n_rows,
                                  n_classes=n_classes)
    data_te = cellGraphDataset(metadatafile=settings['metadata_test'],
                               path_col=settings['path_col'],
                               class_col=settings['class_col'],
                               colList=settings['colList'],
                               colNormalisation=settings['colNormalisation'],
                               colListNewNames=settings['colListNewNames'],
                               micronsPerPixel=settings['micronsPerPixel'],
                               knn=settings['knn'],
                               max_edge_length=settings['max_edge_length'],
                               load_n_rows=load_n_rows,
                               n_classes=n_classes)
    if settings['developmentrun'] is False:
        pickle.dump(data_tr_va, open(settings['tmpGraphDataFile_tr_va'], "wb"))
        pickle.dump(data_te, open(settings['tmpGraphDataFile_te'], "wb"))

# Train/valid/test split
idxs = np.random.permutation(len(data_tr_va))
split_va = int(0.7 * len(data_tr_va))
idx_tr, idx_va = np.split(idxs, [split_va])
data_tr = data_tr_va[idx_tr]
data_va = data_tr_va[idx_va]
data_te = data_te

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=settings['batch_size'], epochs=settings['epochs'])
loader_va = DisjointLoader(data_va, batch_size=settings['batch_size'])
loader_te = DisjointLoader(data_te, batch_size=settings['batch_size'])

labelcounts_va = {}
for la in np.unique(data_va.labels):
    labelcounts_va[la] = data_va.labels.count(la)
print("Label counts validation:", labelcounts_va)

labelcounts_te = {}
for la in np.unique(data_te.labels):
    labelcounts_te[la] = data_te.labels.count(la)
print("Label counts test:", labelcounts_te)

################################################################################
# Build model
################################################################################

# model = Net(n_labels=data_tr.n_labels)
model = GeneralGNN(data_tr.n_labels, activation="softmax")
optimizer = Adam(lr=settings['learning_rate'])
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
            loss_fn(target, pred),  # loss
            tf.reduce_mean(categorical_accuracy(target, pred)),  # accuracy
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

history = {'epochnr': [], 'loss': [], 'acc': [], 'balacc': [], 'val_loss': [], 'val_acc': [], 'val_balacc': [], 'epochtime': []}
epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = settings['es_patience']
results = []
t00 = time.time()
for batch in loader_tr:
    t0 = time.time()
    step += 1
    loss, acc = train_step(*batch)
    balacc = 999
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        val_balacc = 999
        history['epochnr'] += [epoch]
        history['loss'] += [float(loss)]
        history['acc'] += [float(acc)]
        history['balacc'] += [balacc]
        history['val_loss'] += [val_loss]
        history['val_acc'] += [val_acc]
        history['val_balacc'] += [val_balacc]
        t1 = time.time()
        history['epochtime'] += [t1-t0]
        print("Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f} - Val balacc: {:.3f} - Time: {:.3f}s".format(
                epoch, *np.mean(results, 0), val_loss, val_acc, val_balacc, t1-t0))

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_balacc = val_balacc
            patience = settings['es_patience']
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {:.3f}, best val_acc: {:.3f}, best val_balacc: {:.3f})".format(best_val_loss, best_val_acc, best_val_balacc))
                break
        results = []

t11 = time.time()
print("Total training time: {:.3f}s".format(t11-t00))

################################################################################
# Evaluate model - validation set
################################################################################
model.set_weights(best_weights)  # Load best model
val_loss, val_acc = evaluate(loader_va)
val_balacc = 999
print("Done. Validation loss: {:.3f}. Validation acc: {:.3f}. Validation balacc: {:.3f}".format(val_loss, val_acc, val_balacc))


################################################################################
# Evaluate model - test set
################################################################################
model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
test_balacc = 999
print("Done. Test loss: {:.3f}. Test acc: {:.3f}. Test balacc: {:.3f}".format(test_loss, test_acc, test_balacc))


################################################################################
# Save results
################################################################################
history['total_trainingtime'] = "{:.3f}".format(t11-t00)
history['best_val_loss'] = val_loss
history['best_val_acc'] = val_acc
history['best_val_balacc'] = val_balacc
history['best_test_loss'] = test_loss
history['best_test_acc'] = test_acc
history['best_test_balacc'] = test_balacc
with open(os.path.join(outfolder, "history.txt"), "w") as file:
    file.write(json.dumps(history))
