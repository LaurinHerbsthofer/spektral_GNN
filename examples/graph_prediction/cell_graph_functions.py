"""
custom data loading functions for cell seg data
"""

import pandas as pd
import matplotlib.pyplot as plt
from spektral.data import Dataset, Graph
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
import time
import networkx as nx
import os
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class Net(Model):
    def __init__(self, n_labels=2):
        super().__init__()
        self.conv1 = GCSConv(32, activation="relu")
        self.conv2 = GCSConv(32, activation="relu")
        self.conv3 = GCSConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(n_labels, activation="softmax")

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

def loadCellSegData(path, colList, micronsPerPixel, add_cols_to_keep=[], colListNewNames=None, colNormalisation=None):
    """
    load cell seg data, fix some naming issues, apply feature scaling and return as data frame
    @param path:
    @param colList:
    @param colNormalisation:
    @param colListNewNames:
    @return:
    """
    df = read_csv_inForm(path, micronsPerPixel, colList, add_cols_to_keep=add_cols_to_keep)
    # apply weighting to the columns
    if colNormalisation is not None:
        for col, normf in zip(colList, colNormalisation):
            df[col] = df[col] / normf
    # give new names to the columns
    renameDict = dict(zip(colList, colListNewNames))
    df = df.rename(columns=renameDict)
    return df


def read_csv_inForm(filename, micronsPerPixelOrig, colList=None, add_cols_to_keep=[]):
    """
    read a cell seg file as created by Perkin Elmers Inform software
    use colList to read only columns that you desire
    X and Y cell position column names need to be "X" and "Y" for further processing
    @param filename:
    @param micronsPerPixelOrig:
    @param colList:
    @param add_cols_to_keep:
    @return:
    """
    if (colList is None):
        df = pd.read_csv(filename, sep='\t')
    else:
        try:
            # use this for the "conventional" Fusion CCRR column names
            # for Fusion CCRR data, these coordinates are alreyd in microns
            loadCols = colList + ["Cell X Position", "Cell Y Position"] + add_cols_to_keep  # , "Cell ID"]
            df = pd.read_csv(filename, sep='\t', usecols=loadCols)  # , index_col="Cell ID")
        except:
            # for some inform output files it is likely that the column names have a suffix
            # if you want to use the columns that have the same names as the normal marker
            # columns but are following by a suffix "(Normalized Counts, Total Weight)" the use the following code
            # and remove the "return 1" statement above
            # CCRR data, patient IDs: 1016 1018 1031 1032 1034 1049 1052
            print("> Column names not as expected for filename:")
            print("> " + filename)
            # print("> Skipping this file")
            # return 1
            print("> Trying to use the column name with suffix...")
            print("> WARNING: these files have XY coordinates in pixel, not mu. Converting before proceeding.")
            try:
                # fix the fluorescent colnames and cell shape colnames
                loadCols = colList.copy()
                for i, c in enumerate(loadCols):
                    if c.endswith('Area (sq microns)'):
                        loadCols[i] = c.replace('(sq microns)', '(pixels)')
                    elif c.endswith('Ratio'):
                        pass
                    else:
                        loadCols[i] = c + " (Normalized Counts, Total Weighting)"

                # add the remaining colnames
                loadCols = loadCols + ["Cell X Position", "Cell Y Position"] + add_cols_to_keep  # colnames in the file
                renameCols = colList + ["Cell X Position", "Cell Y Position"] + add_cols_to_keep  # desired colnames
                df = pd.read_csv(filename, sep='\t', usecols=loadCols)

                # pay attention here! columns in df may not be in the order you expect since they are sorted alphabetically!
                renameDict = dict(zip(loadCols, renameCols))
                # renameDict = {}
                # for origname, newname in zip(loadCols, renameCols):
                #     renameDict[origname] = newname
                df.rename(columns=renameDict, inplace=True)
                # if this worked indeed, then we need to transform the area pixel feature values to sq micron features
                areaFeats = [x for x in loadCols if x.endswith('Area (sq microns)')]
                for c in areaFeats:
                    df[c] = df[c] * micronsPerPixelOrig**2
                    print("Transformed area feature from pixels to sq microns.")
                # these files for some reason have the XY coordinates in pixels, rather than in mu, so convert them
                df['Cell X Position'] = df['Cell X Position'] * micronsPerPixelOrig
                df['Cell Y Position'] = df['Cell Y Position'] * micronsPerPixelOrig
            except:
                print("> WARNING: Column names not as expected for filename:")
                print("> " + filename)
                print("> WARNING: SKIPPING THIS FILE")
                return 1

    # rename position columns to proper output format
    df.rename(columns={"Cell X Position": "X",
                       "Cell Y Position": "Y"}, inplace=True)
    return df



def assignEdgeLenghts(G):
    """
    for a networkx graph with position arguments for nodes, calculate edge lengths and assign as edge attributes
    @param G:
    @return:
    """
    # positions_dict = nx.get_node_attributes(G, 'pos')  # retrieve positions again
    edge_lengths = {}
    for edge in G.edges():
        node_a = G.nodes()[edge[0]]['pos']
        node_b = G.nodes()[edge[1]]['pos']
        length = ( (node_a[0]-node_b[0])**2 + (node_a[1]-node_b[1])**2 ) ** 0.5
        edge_lengths[edge] = round(length, 2)
    nx.set_edge_attributes(G, edge_lengths, name='length')
    return G


def pruneEdges(G, maxlength=100):
    """
    given a networkx graph with assigned edge attribute "length", remove edges that are longer than given threshold
    @param G:
    @return:
    """
    edge_lengths = nx.get_edge_attributes(G, 'length')
    n = 0
    for edge in G.edges():
        if edge_lengths[edge] > maxlength:
            n += 1
            G.remove_edge(*edge)
    # print("Removed {} edges".format(n))
    return G



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

def plotNXgraph(G, positions_dict, node_size=1, scale_nodes=1, node_color=None, edgelabelname=None, edgewidth=1.0):
    """
    plot a networkx graph together with some edge labels
    @param G:
    @return:
    """
    fig = plt.subplots(figsize=(20, 12))
    node_sizes = [G.nodes()[x][node_size] * scale_nodes for x in G.nodes()] if type(node_size) is str else node_size

    if node_color is None:
        # uniform color nodes
        nx.draw(G, positions_dict, node_size=node_sizes, width=edgewidth)
    else:
        # node color scales with a specified node property
        node_colors = [G.nodes()[x][node_color] for x in G.nodes()]
        nx.draw(G, positions_dict, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.hsv, width=edgewidth)

    # add edge labels (only recommended for testing purposes
    if edgelabelname is None:
        edge_labels = nx.get_edge_attributes(G, edgelabelname)
        nx.draw_networkx_edge_labels(G, pos=positions_dict, edge_labels=edge_labels, font_size=8)

    ax = plt.gca()
    ax.invert_yaxis()
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
    return



################################################################################
# Load data
################################################################################
# class MyDataset(Dataset):
#     """
#     A dataset of random colored graphs.
#     The task is to classify each graph with the color which occurs the most in
#     its nodes.
#     The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
#     nodes connected with probability `p`.
#     """
#
#     def __init__(self, n_samples, n_colors=3, n_min=10, n_max=100, p=0.1, **kwargs):
#         self.n_samples = n_samples
#         self.n_colors = n_colors
#         self.n_min = n_min
#         self.n_max = n_max
#         self.p = p
#         super().__init__(**kwargs)
#
#     def read(self):
#         def make_graph():
#             n = np.random.randint(self.n_min, self.n_max)
#             colors = np.random.randint(0, self.n_colors, size=n)
#
#             # Node features
#             x = np.zeros((n, self.n_colors))
#             x[np.arange(n), colors] = 1
#
#             # Edges
#             a = np.random.rand(n, n) <= self.p
#             a = np.maximum(a, a.T).astype(int)
#             a = sp.csr_matrix(a)
#
#             # Labels
#             y = np.zeros((self.n_colors,))
#             color_counts = x.sum(0)
#             y[np.argmax(color_counts)] = 1
#
#             return Graph(x=x, a=a, y=y)
#
#         # We must return a list of Graph objects
#         return [make_graph() for _ in range(self.n_samples)]
#
#
# class MyDataset2(Dataset):
#     """
#     A dataset of KNN-graphs with multiple features per node (i.e. color values).
#     The task is to classify each graph with the color which occurs the most in
#     its nodes.
#     The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
#     nodes connected with k-nearest neighbors with a miximum length.
#     """
#
#     def __init__(self, n_samples, n_colors=3, n_min=10, n_max=100, knn=5, max_edge_length=None, **kwargs):
#         self.n_samples = n_samples  # how many graphs to create
#         self.n_colors = n_colors  # how many "color" features each node should have
#         self.n_min = n_min  # minimum number of nodes of each graph
#         self.n_max = n_max  # maximum number of nodes of each graph
#         self.knn = knn  # how many edges each node should have
#         self.max_edge_length = max_edge_length  # maximum allowed edge lengths
#         super().__init__(**kwargs)
#
#     def read(self):
#         def make_graph(graphnumber):
#             print("Loading data for graph nr. {}".format(graphnumber))
#             n = np.random.randint(self.n_min, self.n_max)  # number of nodes
#
#             # Node features continuous
#             x = np.random.rand(n, self.n_colors)  # each node gets random color values for each available color
#
#             # create random XY positions for the nodes
#             nodepos = pd.DataFrame({'X': np.random.rand(n), 'Y': np.random.rand(n)})  # X, Y
#
#             # Edges (without weights) represented as adjacency matrix
#             a = kneighbors_graph(nodepos, self.knn)
#             # n_edges_orig = a.count_nonzero()
#             # print("Original knn edges: {}".format(n_edges_orig))
#
#             # get edge lengths and prune edges that are too long
#             n_edge_features = 1  # we only use the length of the edge as edge feature
#             e = np.zeros((n, n, n_edge_features), dtype=float)
#             n_pruned_edges = 0
#             for (i, j) in zip(a.nonzero()[0], a.nonzero()[1]):
#                 n1 = nodepos.iloc[i]
#                 n2 = nodepos.iloc[j]
#                 elength = ((n1['X'] - n2['X'])**2 + (n1['Y'] - n2['Y'])**2) ** 0.5
#                 if self.max_edge_length is None or elength <= self.max_edge_length:
#                     e[i, j, 0] = elength
#                 else:
#                     n_pruned_edges += 1
#                     e[i, j, 0] = 0.
#                     a[i, j] = 0.
#             # print("pruned eges: {}".format(n_pruned_edges))
#             # n_edges = a.count_nonzero()
#             # print("remaining edges: {}".format(n_edges))
#             # plotSKlearnGraph(a, nodepos, dotsize=3)
#             a = sp.csr_matrix(a)
#
#             # Labels, count the number of maximum sum of color values
#             y = np.zeros((self.n_colors,))
#             color_counts = x.sum(0)
#             y[np.argmax(color_counts)] = 1
#
#             return Graph(x=x, a=a, e=None, y=y)
#
#         # We must return a list of Graph objects
#         return [make_graph(i) for i in range(self.n_samples)]


class cellGraphDataset(Dataset):
    """
    A dataset of KNN-graphs from cell seg data with multiple features per node (i.e. color values).
    The task is to classify each graph with a clinical feature
    The graphs have `n_colors` colors, of at least `n_min` and at most `n_max`
    nodes connected with k-nearest neighbors with a miximum length.
    """

    def __init__(self, metadatafile=None, path_col='cellSegFile', class_col='fiveYearRelapse',
                 micronsPerPixel=None, colList=None, colNormalisation=None, colListNewNames=None,
                 knn=5, max_edge_length=None, load_n_rows=None, n_classes=None, verbosity=1,
                 logfile=None, **kwargs):
        self.path_col = path_col
        self.class_col = class_col
        self.colList = colList
        self.colNormalisation = colNormalisation
        self.colListNewNames = colListNewNames
        self.micronsPerPixel = micronsPerPixel
        self.load_n_rows = load_n_rows  # use this for development and testing to only load a few data rows
        self.verbosity = verbosity
        self.logfile = logfile

        self.metadata = pd.read_csv(metadatafile, sep=";", usecols=[self.path_col, self.class_col], nrows=self.load_n_rows)  # [dev: nrows=5], a csv file containing sample paths and clinical data
        self.metadata[self.path_col] = [x.replace('/home/laurin/data0/', '/media/data1/DATA/') for x in self.metadata[self.path_col]]
        self.labels = list(self.metadata[self.class_col])

        self.n_classes = len(self.metadata[self.class_col].unique()) if n_classes is None else n_classes
        self.n_samples = len(self.metadata)
        self.knn = knn  # how many edges each node should have
        self.max_edge_length = max_edge_length  # maximum allowed edge lengths
        self.processingTime = []
        super().__init__(**kwargs)


    def read(self):
        def make_graph(path, classlabel, i, n_total):
            if self.verbosity >= 1:
                print("({}/{}) Loading data for sample: {}".format(i+1, n_total, os.path.basename(path)))
            t0 = time.time()
            df = loadCellSegData(path, self.colList, self.micronsPerPixel,
                                 colListNewNames=self.colListNewNames,
                                 colNormalisation=self.colNormalisation)
            n = len(df)

            # Node features
            x = np.array(df[self.colListNewNames])

            # Edges (without weights) represented as adjacency matrix
            if self.verbosity >= 2:
                print(" > Nodes loaded: {}".format(n))
                print(" > Creating graph and pruning edges...")
            a = kneighbors_graph(df[['X', 'Y']], self.knn)
            # this is somehow not reliable
            # n_edges_orig = int(a.count_nonzero() / 2)  #
            # print(" > Original knn edges: {}".format(n_edges_orig))


            # # OLD VERSION: slow processing
            # # get edge lengths and prune edges that are too long
            # n_edge_features = 1  # we only use the length of the edge as edge feature
            # e = np.zeros((n, n, n_edge_features), dtype=float)
            # n_pruned_edges = 0
            # # this is rather slow...
            # for (i, j) in zip(a.nonzero()[0], a.nonzero()[1]):
            #     n1 = df[['X', 'Y']].iloc[i]
            #     n2 = df[['X', 'Y']].iloc[j]
            #     elength = ((n1['X'] - n2['X'])**2 + (n1['Y'] - n2['Y'])**2) ** 0.5
            #     if self.max_edge_length is None or elength <= self.max_edge_length:
            #         e[i, j, 0] = elength
            #     else:
            #         n_pruned_edges += 1
            #         e[i, j, 0] = 0.
            #         a[i, j] = 0.
            # # e = e # note that the edge weight for the graph should be some form of inverse of the length
            # print(" > Pruned eges: {}".format(n_pruned_edges))
            # n_edges = a.count_nonzero()
            # print(" > Remaining edges: {}".format(n_edges))
            # # plotSKlearnGraph(a, df[['X', 'Y']], dotsize=3)
            # a = sp.csr_matrix(a)

            # # NEW VERSION: faster
            G = nx.to_networkx_graph(a)
            n_edges_orig = len(G.edges())
            if self.verbosity >= 2:
                print(" > Original knn edges: {}".format(n_edges_orig))
            # specify the node positions
            positions_dict = {k: (x, y) for k, x, y in zip(list(G.nodes()), df['X'], df['Y'])}
            nx.set_node_attributes(G, positions_dict, name='pos')  # set positions
            for col in self.colListNewNames:
                nx.set_node_attributes(G, {k: t for k, t in zip(list(G.nodes()), df[col])}, name=col)
            G = assignEdgeLenghts(G)
            G = pruneEdges(G, maxlength=self.max_edge_length)
            if self.verbosity >= 2:
                print(" > Pruned eges: {}".format(n_edges_orig-len(G.edges())))
            n_edges = a.count_nonzero()
            if self.verbosity >= 2:
                print(" > Remaining edges: {}".format(len(G.edges())))
            a = nx.adjacency_matrix(G)
            a = sp.csr_matrix(a)
            # e = nx.adjacency_matrix(G, weight='length')  # note that the edge weight for the graph should be some form of inverse of the length
            # plotting for development
            # positions_dict = {k: (x, y) for k, x, y in zip(list(G.nodes()), df['X'], df['Y'])}
            # plotNXgraph(G, positions_dict, node_size=10)



            # Labels, count the number of maximum sum of color values
            y = np.zeros((self.n_classes,))
            # color_counts = x.sum(0)
            # y[np.argmax(color_counts)] = 1
            y[int(classlabel)] = 1
            t1 = time.time()
            if self.verbosity >= 2:
                print(" > Creating graphs took {:.3f}s".format(t1 - t0))
            self.processingTime += [t1 - t0]
            return Graph(x=x, a=a, e=None, y=y)

        # We must return a list of Graph objects
        t0 = time.time()
        graph_list = [make_graph(path, label, i, len(self.metadata)) for i, (path, label) in enumerate(zip(self.metadata[self.path_col], self.metadata[self.class_col]))]
        t1 = time.time()
        print("Creating all graphs took {:.3f}s".format(t1 - t0))
        df = pd.DataFrame({'image': self.metadata[self.path_col],
                           'processingTime': self.processingTime})
        df.to_csv(self.logfile, sep=";")
        return graph_list


def plotTrainHistory(history, outfolder, historyKeys=['loss', 'acc']):
    '''

    @param train_history:
    @param dataVars:
    @param historyKeys: ['loss', 'acc', 'balacc', 'f1score', 'precision', 'recall', 'mcor']
    @return:
    '''
    '''
    plot the relevant variables from the keras training history
    adds moving average and error estimate for val set for final state
    '''
    df = pd.DataFrame(history)
    windowsize = max(10, len(df) // 10)  # make windowsize 10 percent of epochs, limit windowsize to be no smaller than 10
    flat_epochs = max(10, len(df) // 10)  # take last 10 percent as region to average over, limit to be no smaller than 10
    stats = {}

    for k in historyKeys:
        plt.figure(figsize=(6, 6))

        # train set
        rolling_mean = df[k].rolling(window=windowsize, center=True, win_type='triang').mean()
        df_flat = df[k][len(df) - flat_epochs:len(df)]
        avg = df_flat.mean()
        std = df_flat.std()
        plt.plot(df.index.astype(int), df[k], color='steelblue', alpha=0.5, label=k)
        plt.plot(df.index.astype(int), rolling_mean, color='steelblue', label=k + ' AVG')

        # val set
        kval = 'val_' + k
        rolling_mean = df[kval].rolling(window=windowsize, center=True, win_type='triang').mean()
        df_flat = df[kval][len(df) - flat_epochs:len(df)]
        avg = df_flat.mean()
        std = df_flat.std()
        plt.plot(df.index.astype(int), df[kval], color='darkred', alpha=0.5, label=kval)
        plt.plot(df.index.astype(int), rolling_mean, color='darkred', label=kval + ' AVG')

        plt.ylabel(k)
        plt.xlabel('epoch')
        plt.legend(loc='best')
        plt.title("{0} AVG last {1} epochs: {2:0.3f} +/- {3:0.3f}".format(kval, flat_epochs, avg,
                                                                          std))  # rolling_mean[-int(np.ceil(windowsize/2))
        plt.savefig(os.path.join(outfolder, k + '.png'))
        stats[kval + '_mean_last_epochs'] = avg
        stats[kval + '_std_last_epochs'] = std

    return stats



