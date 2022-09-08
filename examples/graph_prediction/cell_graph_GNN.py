"""
This code is based on the custom_dataset.py example provided by spektral. I updated it by using our custom data loader
that loads cell segmentation data, transforms it into a cell graph using kNN with maximum edge length and uses the
GeneralGNN model proposed by [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)

We use metadata files for the input data seperately for training and test sets to control what goes onto the test set.
Validation splits are random within the test set.

We use pickle to save and load the data set after it's converted to graphs to massivle save data loading times if the model
is rerun multiple times. Make sure to delete these temporary binary files if you want to create the data again from scratch
or with different parameters!

All settings are specified in the settings.yaml file which also offers a "developmentrun" option that loads only a small
number of samples.

Note that the current version of this code only supports binary classification since some balanced accuracy
calculations are not implemented in a more general way than the 2-class problem.

Launch this code from the "venv" virtualenv within this project to run it on the CPU. However, this code can also be
launched from the C2G virtualenv "venvTF24" for GPU support, but using the GeneralGNN model likely requires
an update to sklearn (or scipy). I did not yet do this to avoid breaking the venvTF24 that is used for other C2G classifiers
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from cell_graph_functions import *
import pickle
import yaml
import datetime
import json
from sklearn.metrics import balanced_accuracy_score

from spektral.models import GeneralGNN
from spektral.data import DisjointLoader

################################################################################
# Config
################################################################################

with open('settings.yaml') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outfolder = os.path.join(settings['outfolder'], timestamp + "_" + settings['modelrun_name'])
    os.makedirs(outfolder)
    if settings['developmentrun']:
        always_load_data_anew = True  # [dev: True]
        load_n_rows = settings['load_n_rows']
        n_classes = settings['n_classes']
    else:
        always_load_data_anew = False
        load_n_rows = None
        n_classes = None

t0 = time.time()
if os.path.exists(settings['tmpGraphDataFile_tr_va']) and os.path.exists(settings['tmpGraphDataFile_te']) and always_load_data_anew is False:
    print("Loading graph data set from file ...")
    data_existed = True
    try:
        data_tr_va = pickle.load(open(settings['tmpGraphDataFile_tr_va'], "rb"))
        print(" > Loaded {} training/validation samples.".format(len(data_tr_va)))
        data_te = pickle.load(open(settings['tmpGraphDataFile_te'], "rb"))
        print(" > Loaded {} test samples.".format(len(data_te)))
    except:
        assert False, "You might need to rebuild your graph data (delete the tmp file). Probably you are using a different python environment or have renamed some functions etc."
else:
    print("Creating graph data set ...")
    data_existed = False
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
                                  verbosity=settings['verbosity'],
                                  load_n_rows=load_n_rows,
                                  n_classes=n_classes,
                                  logfile=os.path.join(outfolder, "graph_log_tr_va.csv"))
    data_te = cellGraphDataset(metadatafile=settings['metadata_test'],
                               path_col=settings['path_col'],
                               class_col=settings['class_col'],
                               colList=settings['colList'],
                               colNormalisation=settings['colNormalisation'],
                               colListNewNames=settings['colListNewNames'],
                               micronsPerPixel=settings['micronsPerPixel'],
                               knn=settings['knn'],
                               max_edge_length=settings['max_edge_length'],
                               verbosity=settings['verbosity'],
                               load_n_rows=load_n_rows,
                               n_classes=n_classes,
                               logfile=os.path.join(outfolder, "graph_log_te.csv"))
    if settings['developmentrun'] is False:
        pickle.dump(data_tr_va, open(settings['tmpGraphDataFile_tr_va'], "wb"))
        pickle.dump(data_te, open(settings['tmpGraphDataFile_te'], "wb"))


# THIS DOES NOT WORK ON SUBSETS AS THE .labels WILL NOT CHANGE!
labelcounts_tr_va = {}
for la in np.unique(data_tr_va.labels):
    labelcounts_tr_va[la] = data_tr_va.labels.count(la)
print(" > Label counts training+validation:", labelcounts_tr_va)

labelcounts_te = {}
for la in np.unique(data_te.labels):
    labelcounts_te[la] = data_te.labels.count(la)
print(" > Label counts test:", labelcounts_te)

# Train/valid/test split
idxs = np.random.permutation(len(data_tr_va))
split_va = int((1. - settings['val_set_fraction']) * len(data_tr_va))
idx_tr, idx_va = np.split(idxs, [split_va])
data_tr = data_tr_va[idx_tr]
data_va = data_tr_va[idx_va]
data_te = data_te

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=settings['batch_size'], epochs=settings['epochs'])
loader_va = DisjointLoader(data_va, batch_size=settings['batch_size'])
loader_te = DisjointLoader(data_te, batch_size=settings['batch_size'])

t1 = time.time()
datapreptime = t1 - t0
print(" > Data preparation took: {:.3f}\n".format(datapreptime))

################################################################################
# Build model
################################################################################

assert data_tr.n_labels <= 2, "Some code (especially for balanced accuracy) currently only works for binary classification!"

# model = Net(n_labels=data_tr.n_labels)
model = GeneralGNN(data_tr.n_labels,
                   activation=settings['activation'],
                   hidden=settings['hidden'],
                   message_passing=settings['message_passing'],
                   pre_process=settings['pre_process'],
                   post_process=settings['post_process'],
                   connectivity=settings['connectivity'],
                   batch_norm=settings['batch_norm'],
                   dropout=settings['dropout'],
                   aggregate=settings['aggregate'],
                   hidden_activation=settings['hidden_activation'],
                   pool=settings['pool']
                   )
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
    return (loss, acc), target, predictions


def evaluate(loader):
    output = []
    y_target = []
    y_pred_bin = []
    y_pred = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        y_target += [int(x) for x in np.argmax(target, axis=1)]  # list(target[:, 1])
        pred = model(inputs, training=False)
        y_pred += [float(x) for x in pred[:,1]]  # prediction probability of the 1-class, works only for binary classification!
        y_pred_bin += [int(x) for x in np.argmax(pred, axis=1)]  # [float(x) for x in pred[:, 1]]
        outs = (
            loss_fn(target, pred),  # loss
            tf.reduce_mean(categorical_accuracy(target, pred)),  # accuracy
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1]), y_target, y_pred_bin, y_pred


history = {'epochnr': [], 'loss': [], 'acc': [], 'balacc': [], 'val_loss': [], 'val_acc': [], 'val_balacc': [], 'epochtime': []}
epoch = 1
step = 0
best_val_loss = np.inf
best_weights = None
patience = settings['es_patience']
results = []
t00 = time.time()
print("\n++++++++++++++\nStart training\n++++++++++++++\n")
epochbegintime = time.time()
for batch in loader_tr:
    timesinceepochstart = time.time() - epochbegintime
    step += 1
    remaining_steps = loader_tr.steps_per_epoch - step
    if step == 1:
        ETA = None
        print("Epoch {} - Step {}/{} ...".format(epoch, step, loader_tr.steps_per_epoch), end='\r')
    else:
        ETA = remaining_steps * timesinceepochstart / (step-1)
        print("Epoch {} - Step {}/{} - Step time: {:.1f}s - ETA: {}s ...".format(epoch, step, loader_tr.steps_per_epoch, laststeptime, int(ETA)), end='\r')

    t0step = time.time()
    (loss, acc), y_target, y_pred = train_step(*batch)
    y_target = np.argmax(y_target, axis=1).astype(int)
    y_pred_bin = np.argmax(y_pred, axis=1).astype(int)
    y_pred = np.array(y_pred)[:,1].tolist()  # works only for binary classificaiton!
    balacc = balanced_accuracy_score(y_target, y_pred_bin)
    results.append((loss, acc))
    t1step = time.time()
    laststeptime = t1step - t0step
    if step == loader_tr.steps_per_epoch:
        print("Epoch {} - Evaluating validation set ................................................".format(epoch), end='\r')
        # Compute validation loss and accuracy
        (val_loss, val_acc), y_target, y_pred_bin, y_pred = evaluate(loader_va)
        val_balacc = balanced_accuracy_score(y_target, y_pred_bin)
        history['epochnr'] += [epoch]
        history['loss'] += [float(loss)]
        history['acc'] += [float(acc)]
        history['balacc'] += [balacc]
        history['val_loss'] += [val_loss]
        history['val_acc'] += [val_acc]
        history['val_balacc'] += [val_balacc]
        epochendtime = time.time()  # end time of epoch
        history['epochtime'] += [epochendtime-epochbegintime]
        print("Epoch {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f} - Val balacc: {:.3f} - Time: {:.3f}s".format(
                epoch, *np.mean(results, 0), val_loss, val_acc, val_balacc, epochendtime-epochbegintime))

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_balacc = val_balacc
            patience = settings['es_patience']
            print(" > New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print(" +++ Early stopping (best val_loss: {:.3f}, best val_acc: {:.3f}, best val_balacc: {:.3f})".format(best_val_loss, best_val_acc, best_val_balacc))
                break
        results = []

        # advance to next epoch
        step = 0
        epoch += 1
        epochbegintime = time.time()

t11 = time.time()
print("Total training time: {:.3f}s".format(t11-t00))


# trainhistory = {key: history[key] for key in ['epochnr', 'loss', 'val_loss', 'acc', 'val_acc', 'balacc', 'val_balacc']}
trainhistory = history
plotTrainHistory(trainhistory, outfolder, historyKeys=['loss', 'acc', 'balacc'])

################################################################################
# Evaluate model - validation set
################################################################################
model.set_weights(best_weights)  # Load best model
(val_loss, val_acc), y_val_target, y_val_pred_bin, y_val_pred = evaluate(loader_va)
val_balacc = balanced_accuracy_score(y_val_target, y_val_pred_bin)
print("Done. Validation loss: {:.3f}. Validation acc: {:.3f}. Validation balacc: {:.3f}".format(val_loss, val_acc, val_balacc))


################################################################################
# Evaluate model - test set
################################################################################
model.set_weights(best_weights)  # Load best model
(test_loss, test_acc), y_test_target, y_test_pred_bin, y_test_pred = evaluate(loader_te)
test_balacc = balanced_accuracy_score(y_test_target, y_test_pred_bin)
print("Done. Test loss: {:.3f}. Test acc: {:.3f}. Test balacc: {:.3f}".format(test_loss, test_acc, test_balacc))


################################################################################
# Save results
################################################################################
history['total_datapreptime'] = datapreptime
history['data_existed'] = data_existed
history['total_trainingtime'] = "{:.3f}".format(t11-t00)
history['best_val_loss'] = val_loss
history['best_val_acc'] = val_acc
history['best_val_balacc'] = val_balacc
history['best_val_y_target'] = y_val_target
history['best_val_y_pred_bin'] = y_val_pred_bin
history['best_val_y_pred'] = y_val_pred
history['best_test_loss'] = test_loss
history['best_test_acc'] = test_acc
history['best_test_balacc'] = test_balacc
history['best_test_y_target'] = y_test_target
history['best_test_y_pred_bin'] = y_test_pred_bin
history['best_test_y_pred'] = y_test_pred
with open(os.path.join(outfolder, "history.json"), "w") as file:
    file.write(json.dumps(history))
with open(os.path.join(outfolder, "settings.json"), "w") as file:
    file.write(json.dumps(settings))
pickle.dump(best_weights, open(os.path.join(outfolder, "best_model_weights.pickle"), "wb"))

print("\n+++++++++\nAll done.\n+++++++++\n")

