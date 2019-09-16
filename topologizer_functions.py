from __future__ import division
import numpy as np
from sklearn import preprocessing
import json
from collections import defaultdict
from statistics import mode
from collections import defaultdict
from datetime import datetime
import inspect
import itertools
import os
import sys
import warnings
from matplotlib import colors
from jinja2 import Environment, FileSystemLoader, Template
import numpy as np
from sklearn import cluster, preprocessing, manifold, decomposition
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.spatial import distance
from scipy.sparse import issparse
from statistics import mode, StatisticsError
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
import networkx as nx
from matplotlib import pyplot
import kmapper as km
import scipy.cluster.hierarchy as sch
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.spatial.distance
import scipy.stats
import sklearn.cluster
import sklearn.metrics.pairwise
from jinja2 import Environment, FileSystemLoader, Template
import os
from ast import literal_eval
from scipy import stats
from kmapper.visuals import (
    init_color_function,
    format_meta,
    format_mapper_data,
    build_histogram,
    graph_data_distribution,
)
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find
from scipy.sparse.linalg import eigs
import pandas as pd

#### Note: the diffusion map code is adapted from palantir (https://github.com/dpeerlab/Palantir)

def run_diffusion_maps(data_df, n_components=10, knn=30, n_jobs=-1):
    """Run Diffusion maps using the adaptive anisotropic kernel
    :param data_df: PCA projections of the data or adjancency matrix
    :param n_components: Number of diffusion components
    :return: Diffusion components, corresponding eigen values and the diffusion operator
    """

    # Determine the kernel
    N = data_df.shape[0]
    if not issparse(data_df):
        print('Determing nearest neighbor graph...')
        nbrs = NearestNeighbors(n_neighbors=int(knn), metric='euclidean',
                                n_jobs=n_jobs).fit(data_df.values)
        kNN = nbrs.kneighbors_graph(data_df.values, mode='distance')

        # Adaptive k
        adaptive_k = int(np.floor(knn / 3))
        nbrs = NearestNeighbors(n_neighbors=int(adaptive_k),
                                metric='euclidean', n_jobs=n_jobs).fit(data_df.values)
        adaptive_std = nbrs.kneighbors_graph(
            data_df.values, mode='distance').max(axis=1)
        adaptive_std = np.ravel(adaptive_std.todense())

        # Kernel
        x, y, dists = find(kNN)

        # X, y specific stds
        dists = dists / adaptive_std[x]
        W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

        # Diffusion components
        kernel = W + W.T
    else:
        kernel = data_df

    # Markov
    D = np.ravel(kernel.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
    # Eigendecomposition
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000)
    D = np.real(D)
    V = np.real(V)
    inds = np.argsort(D)[::-1]
    D = D[inds]
    V = V[:, inds]

    # Normalize
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

    # Create are results dictionary
    res = {'T': T, 'EigenVectors': V, 'EigenValues': D}
    res['EigenVectors'] = pd.DataFrame(res['EigenVectors'])
    if not issparse(data_df):
        res['EigenVectors'].index = data_df.index
    res['EigenValues'] = pd.Series(res['EigenValues'])

    return res

## Note: the functions below are adapted from kmapper (https://github.com/scikit-tda/kepler-mapper)

def set_km_color_map(bin_colors,ncols):
    set_cols = []
    if ncols == 1:
        set_cols.append([1.0,'rgb('+ str(int(bin_colors[0][0]*255)) + ',' + str(int(bin_colors[0][1]*255)) + ',' + str(int(bin_colors[0][2]*255)) + ')'])
    else:
        for i in range(ncols):
            set_cols.append([i/(ncols-1),'rgb('+ str(int(bin_colors[i][0]*255)) + ',' + str(int(bin_colors[i][1]*255)) + ',' + str(int(bin_colors[i][2]*255)) + ')'])
    set_cols.insert(0,[-1.0,'rgb(211,211,211)'])
    
    km.visuals.colorscale_default = set_cols

set1 = pyplot.get_cmap('Set1')
bin_colors = set1([0,1,2,3,4,5,6,7,8,9,10])
set1_cols = []
nbins=10
for i in range(nbins+1):
    set1_cols.append([i/nbins,'rgb('+ str(int(bin_colors[i][0]*255)) + ',' + str(int(bin_colors[i][1]*255)) + ',' + str(int(bin_colors[i][2]*255)) + ')'])
#km.visuals.colorscale_default = set1_cols
set_km_color_map(set1([0,1,2,3,4,5,6,7,8,9,10]),10)

def _tooltip_components(
    member_ids,
    X,
    X_names,
    lens,
    lens_names,
    color_function,
    node_ID,
    colorscale,
    nbins=10,
):
    projection_stats = km.visuals._format_projection_statistics(member_ids, lens, lens_names)
    cluster_stats = km.visuals._format_cluster_statistics(member_ids, X, X_names)

    member_histogram = build_histogram(
        color_function[member_ids], colorscale=colorscale, nbins=nbins
    )

    return projection_stats, cluster_stats, member_histogram

def _format_tooltip(
    env,
    member_ids,
    custom_tooltips,
    X,
    X_names,
    lens,
    lens_names,
    color_function,
    node_ID,
    nbins,
):
    # TODO: Allow customization in the form of aggregate per node and per entry in node.
    # TODO: Allow users to turn off tooltip completely.

    custom_tooltips = (
        custom_tooltips[member_ids] if custom_tooltips is not None else member_ids
    )

    # list will render better than numpy arrays
    custom_tooltips = list(custom_tooltips)

    colorscale = km.visuals.colorscale_default

    projection_stats, cluster_stats, histogram = _tooltip_components(
        member_ids,
        X,
        X_names,
        lens,
        lens_names,
        color_function,
        node_ID,
        colorscale,
        nbins,
    )

    tooltip = env.get_template("cluster_tooltip.html").render(
        projection_stats=projection_stats,
        cluster_stats=cluster_stats,
        custom_tooltips=custom_tooltips,
        histogram=histogram,
        dist_label="Member",
        node_id=node_ID,
    )

    return tooltip

def graph_data_distribution(graph, color_function, colorscale, nbins=10):

    node_averages = []
    for node_id, member_ids in graph["nodes"].items():
        member_colors = color_function[member_ids]
        node_averages.append(np.mean(member_colors))

    histogram = build_histogram(node_averages, colorscale=colorscale, nbins=nbins)

    return histogram

def build_histogram(data, colorscale=None, nbins=10):
    """ Build histogram of data based on values of color_function
    """
    histogram = []

    for i in range(nbins):
        histogram.append({"height": int(0), "perc": int(0), "color": 'rgb(211,211,211)'})

    return histogram

def _color_function(member_ids, color_function):
    color_func_filt = color_function[member_ids]
    #NAs will be converted to negative value
    color_func_filt = color_func_filt[color_func_filt > 0]
    
    if color_func_filt.shape[0] == 0:
        return 0.0
    return stats.mode(color_func_filt)[0][0]

def _size_node(member_ids,color_function,node_size_func="none"):
    num_cells_filt = color_function[member_ids]
    num_cells_filt = np.sum(num_cells_filt > 0)
    if num_cells_filt == 0:
        num_cells_filt=0
    if node_size_func == "none":
        return int(num_cells_filt)
    if node_size_func == "sqrt":
        return int(np.sqrt(num_cells_filt))

def format_mapper_data(
    graph, color_function, X, X_names, lens, lens_names, custom_tooltips, env, nbins=10,node_size_func="none"
):
    # import pdb; pdb.set_trace()
    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
        node_id_to_num[node_id] = i
        c = _color_function(member_ids, color_function)
        t = km.visuals._type_node()
        s = _size_node(member_ids,color_function,node_size_func)
        tt = _format_tooltip(
            env,
            member_ids,
            custom_tooltips,
            X,
            X_names,
            lens,
            lens_names,
            color_function,
            node_id,
            nbins
        )

        n = {
            "id": "",
            "name": node_id,
            "color": c,
            "type": km.visuals._type_node(),
            "size": s,
            "tooltip": tt,
        }

        json_dict["nodes"].append(n)
    for i, (node_id, linked_node_ids) in enumerate(graph["links"].items()):
        for linked_node_id in linked_node_ids:
            l = {
                "source": node_id_to_num[node_id],
                "target": node_id_to_num[linked_node_id],
                "width": km.visuals._size_link_width(graph, node_id, linked_node_id),
            }
            json_dict["links"].append(l)
    return json_dict

def visualize(
        graph,
        color_function=None,
        custom_tooltips=None,
        custom_meta=None,
        path_html="mapper_visualization_output.html",
        title="Kepler Mapper",
        save_file=True,
        X=None,
        X_names=[],
        lens=None,
        lens_names=[],
        show_tooltips=True,
        nbins=10,
        verbose=1,node_size_func="none"
    ):
        """Generate a visualization of the simplicial complex mapper output. Turns the complex dictionary into a HTML/D3.js visualization
        Parameters
        ----------
        graph : dict
            Simplicial complex output from the `map` method.
        path_html : String
            file name for outputing the resulting html.
        custom_meta: dict
            Render (key, value) in the Mapper Summary pane. 
        custom_tooltip: list or array like
            Value to display for each entry in the node. The cluster data pane will display entry for all values in the node. Default is index of data.
        save_file: bool, default is True
            Save file to `path_html`.
        X: numpy arraylike
            If supplied, compute statistics information about the original data source with respect to each node.
        X_names: list of strings
            Names of each variable in `X` to be displayed. If None, then display names by index.
        lens: numpy arraylike
            If supplied, compute statistics of each node based on the projection/lens
        lens_name: list of strings
            Names of each variable in `lens` to be displayed. In None, then display names by index.
        show_tooltips: bool, default is True.
            If false, completely disable tooltips. This is useful when using output in space-tight pages or will display node data in custom ways.
        nbins: int, default is 10
            Number of bins shown in histogram of tooltip color distributions.
        Returns
        --------
        html: string
            Returns the same html that is normally output to `path_html`. Complete graph and data ready for viewing.
        Examples
        ---------
        >>> mapper.visualize(simplicial_complex, path_html="mapper_visualization_output.html",
                            custom_meta={'Data': 'MNIST handwritten digits', 
                                         'Created by': 'Franklin Roosevelt'
                            }, )
        """

        # TODO:
        #   - Make color functions more intuitive. How do they even work?
        #   - Allow multiple color functions that can be toggled on and off.

        if not len(graph["nodes"]) > 0:
            raise Exception(
                "Visualize requires a mapper with more than 0 nodes. \nIt is possible that the constructed mapper could have been constructed with bad parameters. This occasionally happens when using the default clustering algorithm. Try changing `eps` or `min_samples` in the DBSCAN clustering algorithm."
            )

        # Find the module absolute path and locate templates
        module_root = os.path.join(os.path.dirname(km.__file__), "templates")
        env = Environment(loader=FileSystemLoader(module_root))
        # Color function is a vector of colors?
        color_function = init_color_function(graph, color_function)

        mapper_data = format_mapper_data(
            graph, color_function, X, X_names, lens, lens_names, custom_tooltips, env, nbins, node_size_func
        )

        colorscale = km.visuals.colorscale_default
        
        histogram = graph_data_distribution(graph, color_function, colorscale)

        mapper_summary = format_meta(graph, custom_meta)

        # Find the absolute module path and the static files
        #js_path = os.path.join(os.path.dirname(km.__file__), "static", "kmapper.js")
        js_path = "kmapper.js"
        with open(js_path, "r") as f:
            js_text = f.read()

        css_path = os.path.join(os.path.dirname(km.__file__), "static", "style.css")
        with open(css_path, "r") as f:
            css_text = f.read()

        # Render the Jinja template, filling fields as appropriate
        template = env.get_template("base.html").render(
            title=title,
            mapper_summary=mapper_summary,
            histogram=histogram,
            dist_label="Node",
            mapper_data=mapper_data,
            colorscale=colorscale,
            js_text=js_text,
            css_text=css_text,
            show_tooltips=True,
        )

        if save_file:
            with open(path_html, "wb") as outfile:
                if verbose > 0:
                    print("Wrote visualization to: %s" % (path_html))
                outfile.write(template.encode("utf-8"))

        return template

def hierarchical_clustering(mat, linkage='average', cluster_distance=False, labels=None, thresh=0.25,metric='euclidean'):
    """
    Performs hierarchical clustering based on distance matrix 'mat' using the method specified by 'method'.
    Optional argument 'labels' may specify a list of labels. If cluster_distance is True, the clustering is
    performed on the distance matrix using euclidean distance. Otherwise, mat specifies the distance matrix for
    clustering. Adapted from
    http://stackoverflow.com/questions/7664826/how-to-get-flat-clustering-corresponding-to-color-clusters-in-the-dendrogram-cre
    Not subjected to copyright.
    """
    if mat.shape[0] == 1:
        return np.array([1])
    D = np.array(mat)
    if cluster_distance:
        Dtriangle = scipy.spatial.distance.squareform(D)
    else:
        Dtriangle = scipy.spatial.distance.pdist(D, metric='euclidean')
    Y = sch.linkage(Dtriangle, method=linkage)
    #Z1 = sch.dendrogram(Y, orientation='right', color_threshold=thresh*max(Y[:, 2]))
    Z1 = sch.fcluster(Y,thresh*max(Y[:, 2]),'distance')
    return Z1

class AgglomerativeHierarchical(BaseEstimator, ClusterMixin):    
    def __init__(self, thresh, linkage='average', metric='euclidean'):
        self.thresh = thresh
        self.linkage = linkage
        self.metric = metric

    def fit(self, X, y=None, sample_weight=None):
        self.labels_ = hierarchical_clustering(X,thresh=self.thresh,linkage=self.linkage,metric=self.metric)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X)
        return self.labels_


class DummyDR(BaseEstimator, TransformerMixin):  
    """Dummy dimensionality reduction algorithm that just returns the data that you pass in."""

    def __init__(self,data=None):
        """
        Dummy initialization
        """
        self.data = data
        
    def fit_transform(self, X):
        """
        Passes your data right back to you.
        """
        return self.data

def to_networkx(graph):
    nodes = graph["nodes"].keys()
    edges = [[start, end] for start, ends in graph["links"].items() for end in ends]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    #nx.set_node_attributes(g, dict(graph["nodes"], "membership"))
    g.add_edges_from(edges)
    return g

def filter_mapper(graph,thresh=3):
    g = to_networkx(graph)
    CCs = nx.algorithms.components.connected_components(g)
    small_CCs = []
    temp = graph
    for c in sorted(CCs, key=len, reverse=True):
        if len(c) < thresh:
            [small_CCs.append(i) for i in list(c)]
    for i in small_CCs:
        if i in temp['nodes'].keys():
            del temp['nodes'][i]
        if i in temp['links'].keys():
            del temp['links'][i]
        if i in temp['meta_nodes'].keys():
            del temp['meta_nodes'][i]
    to_remove = set(small_CCs)
    i=0
    while i < len(temp['simplices']):
        s = set(temp['simplices'][i])
        sd = s - to_remove
        temp['simplices'][i] = list(sd)
        if len(temp['simplices'][i]) == 0:
            del temp['simplices'][i]
        i = i+1
    to_remove2=[]
    for i in temp['links']:
        s = set(temp['links'][i])
        sd = s - to_remove
        temp['links'][i] = list(sd)
        if len(temp['links'][i]) == 0:
            to_remove2.append(i)
    for i in to_remove2:
        del temp['links'][i]
    return temp
