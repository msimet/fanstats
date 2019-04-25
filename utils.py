# utils.py: contains utilities to help analyze and plot AO3 metadata.  Includes both plotting
# routines and methods to compute auto- and cross-correlation matrices.

import urllib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patheffects as path_effects
import pandas as pd
import statsmodels.api as sm
import scipy.interpolate as interpolate
from adjustText import adjust_text

def get_topten_fandoms(db):
    bts_string = list(db.get_data("FandomName", conditions=['FandomName LIKE "%Bangtan%Boys%BTS"'])['FandomName'])[0]

    fandom_data = pd.DataFrame([('Marvel', 'Marvel', 293780),
                                ('Real Person Fiction', 'Real Person Fiction', 212583), 
                                ('Supernatural', 'Supernatural', 201021),
                                ('K-pop', 'K-Pop', 200450),
                                ('Harry Potter - J. K. Rowling', 'Harry Potter - J. K. Rowling', 195963),
                                ('The Avengers - Ambiguous Fandom', 'The Avengers - Ambiguous Fandom', 138462),
                                ('DCU', 'Dcu', 135795),
                                ('Sherlock Holmes & Related Fandoms', 'Sherlock Holmes &Amp; Related Fandoms', 119857),
                                ('Teen Wolf (TV)', 'Teen Wolf (Tv)',100644),
                                ('방탄소년단 | Bangtan Boys | BTS', bts_string, 85740)], 
                               columns=['FandomName', 'DBFandomName', 'ReportedNum'])
    return fandom_data

def renorm_kde_plot(data, ax=None, color=None, bw=None):
    """ Make a KDE plot with Seaborn, and then renormalize the y-axis to be a number instead of a density. """
    if ax==None:
        ax = plt.gca()
    # Make a kernel density estimator smoothed version of the number of posts over time.
    if bw:
        sns.kdeplot(data, shade=True, cut=0, ax=ax, color=color, bw=bw, legend=False)
    else:
        sns.kdeplot(data, shade=True, cut=0, ax=ax, color=color, legend=False)
    # When you make a KDE estimate with Seaborn, it gives you a true density.  However, that's kind of hard for most
    # non-math types to read.  So, I rescale the axis labels so that we get the averaged y-value around that point,
    # instead of the density.  The logarithm stuff is just figuring out a good number of axis ticks.
    xdata = ax.lines[0].get_xdata()
    multiplier = len(data)/(np.sum(ax.lines[0].get_ydata()))/(xdata[1]-xdata[0])
    ylim = ax.get_ylim()[1]*multiplier
    log_ylim = np.log10(ylim)
    ylim /= 10**int(log_ylim)
    if 1<ylim<=2:
        log_ylim -= 1
        ylim *= 10
    ylim += 1
    if ylim < 5:
        new_ytick_labels = 0.5*np.arange(2*int(ylim))
    else:
        new_ytick_labels = np.arange(int(ylim))
    if len(new_ytick_labels)>10:
        new_ytick_labels = new_ytick_labels[::2]
    new_ytick_labels *= 10**int(log_ylim)
    new_ytick_values = new_ytick_labels/multiplier
    ax.set_yticks(new_ytick_values)
    ax.set_yticklabels([int(n) for n in new_ytick_labels])
    return ax

def rescale_axis(ax, axis="x", rescale=1000):
    """ Rescale either the 'x' or 'y' axis by some factor, and alter the axis label to reflect it.
        rescale must be an integer power of 10, 10 <= rescale <= 1000000. """
    relabel = {10: "tens", 100: "hundreds", 1000: "thousands",
               10000: "tens of thousands", 100000: "hundreds of thousands",
               1000000: "millions", 10000000: "tens of millions", 
               100000000: "hundreds of millions", 1000000000: "billions"}
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/rescale))
    if axis=="x":
        label = ax.get_xlabel()
    else:
        label = ax.get_ylabel()
    if label:
        label += " (in {})".format(relabel[rescale])
    if axis=="x":
        ax.xaxis.set_major_formatter(ticks)
        ax.set_xlabel(label)
    else:
        ax.yaxis.set_major_formatter(ticks)
        ax.set_ylabel(label)
    return ax

def add_log_axis_labels(ax, axis='x'):
    """ If log(data) was plotted on the x- or y-axis but you want linear data labels, this function changes them over."""
    if axis=='x' or axis=='both':
        current_ticks = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(int(current_ticks[0]), int(current_ticks[1])+1))
        ticklabels = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(10.**x))
        ax.xaxis.set_major_formatter(ticklabels)
    if axis=='y' or axis=='both':
        current_ticks = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(int(current_ticks[0]), int(current_ticks[1])+1))
        ticklabels = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(10.**x))
        ax.yaxis.set_major_formatter(ticklabels)
    return ax

def format_tag(s):
    """ Take a tag name generated from a URL and possibly lowercased (like "john%20watson*s*sherlock%20holmes")
        and give it some human-readable formatting. """
    s = urllib.parse.unquote(s)
    # Some AO3-specific replacements
    s = u'/'.join(s.split(u'*s*'))
    s = u'.'.join(s.split(u'*d*'))
    s = u"'".join(s.split(u'&#39;'))
    s = u"'".join(s.split(u"&#x27;"))
    return s.title()


def pivot_for_correlation(data, col):
    """ Take a table formatted like [[ID1, tag1], [ID1+1000000000, tag2], [ID2, tag2]]
        and turn it into a table like this:
             tag1 tag2
         ID1    1    1
         ID2  NaN    1
         which presumes my database setup where rather than having a separate table ID and work ID for tables with
         multiple values per ID, we just use a table ID of work ID plus some multiple of 10^9 (greater than the largest
         AO3 work ID number).
    """
    data = data.copy()
    data['true_ID'] = data['ID']%1000000000
    data['ones'] = np.ones(len(data['true_ID']))
    data = data.pivot_table(index='true_ID', columns=col, values='ones')
    data.index.names = ['ID']
    return data
    
def cross_correlate(data0, col0, data1, col1, pivot=True, nworks=None):
    """ Given two data sets and two columns of interest, generate a cross-correlation matrix to plot.
    
    Inputs
    ------
    data0: pandas.DataFrame
        Either a pandas.DataFrame with 'ID' and 'col0' columns, or a pandas.DataFrame that looks like the output
        of the function pivot_for_correlation.
    col0: str
        the column of interest from data0 (only used if pivot=True)
    data1: pandas.DataFrame
        As data0.
    col1: str
        as col0.
    pivot: bool, default True
        Whether to pivot the table using pivot_for_correlation.
    nworks: int, default None
        How many works were in the original data set (None uses len(data0) and len(data1))
        
    Returns
    -------
    new_index0: list
        the indices corresponding to axis 0 of the returned heatmaps (that is, the unique categories in col0).
    new_index1: list
        as new_index0
    heat_matrix: np.array
        the raw counts of how many works have both [item from new_index0] and [item from new_axis1] as a 2d matrix
    norm_heat_matrix: np.array
        heat_matrix minus the expected counts in each bin if all items in each column were randomly distributed
    """
    if not hasattr(pivot, '__iter__'):
        pivot = [pivot, pivot]
    if pivot[0]:
        data0 = pivot_for_correlation(data0, col0)
    if pivot[1]:
        data1 = pivot_for_correlation(data1, col1)
    # This should reset the index if the ID isn't currently the index
    if 'ID' in data1:
        data1 = data1.set_index("ID")
    if 'ID' in data0:
        data0 = data0.set_index("ID")
    if nworks is None:
        lend0 = 1.0*len(data0)
        lend1 = 1.0*len(data1)
        lend = min(lend0, lend1)
    else:
        lend0 = lend1 = lend = nworks
    # Easiest way I know to get these as separate arrays, but a merged index
    merged = data0.merge(data1, how='outer', left_index=True, right_index=True)
    data0 = data0.reindex(merged.index)
    data1 = data1.reindex(merged.index)
    del merged # free up memory
    
    new_index0 = list(data0)
    new_index1 = list(data1)
    heat_matrix = np.zeros((len(new_index0), len(new_index1)))
    for i, idx0 in enumerate(new_index0):
        for j, idx1 in enumerate(new_index1):
            heat_matrix[i,j] += np.sum((data0[idx0] == 1) & (data1[idx1] == 1))
    expected_matrix = np.sum(data0.values==1, axis=0)[:, None]*np.sum(data1.values==1, axis=0)/lend0/lend1*lend
    norm_heat_matrix = heat_matrix/expected_matrix
    # Somehow this reverses x and y indices--so flip them back
    return new_index0, new_index1, heat_matrix.T, norm_heat_matrix.T

def reorder_array(arr, old_order, new_order, axis='both'):
    """ Reorder the rows and/or columns of a rectangular np.array. """
    remapper = np.array([old_order.index(no) for no in new_order])
    if axis=='x' or axis=='both':
        arr = (arr.T[remapper]).T
    if axis=='y' or axis=='both':
        arr = arr[remapper]
    return arr
    
def correlate(data, col, pivot=True, nworks=None):
    """ Return a matrix of the autocorrelation of unique items from a column in a dataset (e.g. how often two tags are
        used together vs separately).
    
    Inputs
    ------
    data: pandas.DataFrame
        Either a pandas.DataFrame with 'ID' and 'col' columns, or a pandas.DataFrame that looks like the output
        of the function pivot_for_correlation.
    col: str
        the column of interest (only used if pivot=True)
    pivot: bool, default True
        Whether to pivot the table using pivot_for_correlation.
    nworks: int, default None
        The number of works in the original data set (default None uses len(data))
        
    Returns
    -------
    new_index: list
        the indices corresponding to each row or column of the returned heatmaps (that is, the unique categories in col).
    heat_matrix: np.array
        the raw counts of how many works have both item i and item j from new_index, or (on the diagonal) how many
        instances of each unique item from new_index appeared.
    norm_heat_matrix: np.array
        heat_matrix minus the expected counts in each bin if all items in the column were randomly distributed
    """
    if pivot:
        data = pivot_for_correlation(data, col)

    new_index = list(data)
    heat_matrix = np.zeros((len(new_index), len(new_index)))
    for i, idx0 in enumerate(new_index):
        for j, idx1 in enumerate(new_index):
            heat_matrix[i,j] += np.sum((data[idx0] == 1) & (data[idx1] == 1))
    sum_of_each = np.sum(data.values==1, axis=0)
    if nworks is None:
        nworks = len(data)
    expected_matrix = sum_of_each*sum_of_each[:, None]/nworks
    norm_heat_matrix = heat_matrix/expected_matrix
    # This doesn't work for the 1:1 terms which aren't really cross-correlations: force to one.
    for i in range(len(new_index)):
        norm_heat_matrix[i,i]=1
    return new_index, heat_matrix, norm_heat_matrix

def write_dataframe(data, fn):
    """ Write all  the data in pd.DataFrame 'data' to the file with filename 'fn' """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        with open(fn, 'w') as f:
            f.write(data.to_string())

def hits_v_kudos_plot(interpolators, labels, all_interpolator, cmap=plt.get_cmap("viridis"), figsize=(12, 8)):
    """ Make a nice plot of all the hits vs kudos trendlines.  interpolators and labels should
        have the same ordering; interpolators should all be functions that return a single value
        for each element in the vector plot_x."""
    fig, ax = plt.subplots(figsize=figsize)#, gridspec_kw={"right": 0.7})
    max_x = []
    max_y = []
    colors = []
    for i, (x,y) in enumerate(interpolators):
        if not np.any(y>0):
            max_x.append(None)
            max_y.append(None)
            colors.append(None)
            continue
        indx = np.argmax(x[y>0])
        max_x.append(x[indx])
        max_y.append(y[indx]/x[indx])
        color = cmap(1.0*i/len(interpolators))
        colors.append(color)
        ax.plot(x, rolling_mean(y/x), color=color)
    x, y = all_interpolator
    indx = np.argmax(x[y>0])
    max_x.append(x[indx])
    max_y.append(y[indx]/x[indx])
    ax.plot(x, rolling_mean(y/x), color=cmap(1.0))
    annotations = []
    for x, y, label, color in zip(max_x, max_y, labels+["All fics"], colors+[cmap(1.0)]):
        if x is not None:
            text = ax.text(x, y, label, color=color)
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                   path_effects.Normal()])
            annotations.append(text)
    adjust_text(annotations)
    ax.set_xlabel("Hits")
    ax.set_ylabel("Kudos/hits ratio")
    #ax.set_ylim((0, ax.get_ylim()[1]))
    return fig

def get_duplicate_parent(arr):
    l = len(arr)
    arr = arr[~arr.isnull()]
    if not arr.any():
        return None
    ret = arr[arr.index].values[0]
    return ret

def deduplicate_parent(arr):
    mask = arr.index.duplicated()
    vals = arr.index[mask].unique()
    for v in vals:
        arr.loc[v, 'ParentName'] = get_duplicate_parent(arr['ParentName'][v])
    return arr[~arr.index.duplicated(keep='first')]

def get_parent_columns(db):
    
    a = db.cur.execute("PRAGMA table_info(Fandoms);").fetchall()
    col_names = [aa[1] for aa in a]
    if 'OriginalFandomName' in col_names:
        return

    parent_mapper = db.get_data("FandomName", "ParentName", "URLName", "ReportedNum").set_index("FandomName")
    parent_mapper['OriginalParentName'] = parent_mapper['ParentName'].copy()
    mask = ~parent_mapper["ParentName"].isnull()
    parent_mapper["ParentName"][mask] = parent_mapper["ParentName"][mask].map(format_tag)
    parent_mapper.index = parent_mapper.index.map(format_tag)
    parent_mapper = deduplicate_parent(parent_mapper)
    mask = parent_mapper.index == parent_mapper['ParentName']
    parent_mapper.loc[mask, 'ParentName'] = None
    parent_mapper.loc[parent_mapper.index=="Rome (Tv 2005)", 'ParentName'] = None
    new_fandoms = set(parent_mapper['ParentName']) - set(parent_mapper.index)
    added_df = pd.DataFrame(dict(FandomName = list(new_fandoms), ParentName = [None]*len(new_fandoms))).set_index("FandomName")
    parent_mapper = pd.concat([parent_mapper, added_df])

    
    
    def get_parent(s):
        i = 0
        while parent_mapper['ParentName'][s]:
            s = parent_mapper['ParentName'][s]
            i += 1
            if i>5:
                raise RuntimeError("Too many iterations for", s)
        return s

    true_parents = parent_mapper.drop('ParentName', axis=1)
    true_parents['ParentName'] = true_parents.index.map(get_parent)

    rows = [(index, row['ParentName'], row['OriginalParentName'], row['URLName'], row['ReportedNum']) 
            for index, row in true_parents.iterrows()]
    if len(db.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='OriginalFandoms'").fetchall()) == 0:
        db.cur.execute("CREATE TABLE OriginalFandoms AS SELECT * FROM Fandoms;")
    db.cur.execute("DROP TABLE Fandoms;")
    db.cur.execute("CREATE TABLE Fandoms(OriginalFandomName TEXT PRIMARY KEY, FandomName TEXT, ParentName TEXT, URLName TEXT, ReportedNum INT);")
    db.cur.executemany("INSERT INTO Fandoms VALUES(?, ?, ?, ?, ?)", rows)
    db.con.commit()

def redo_work_columns(db):
    data = pd.DataFrame(db.cur.execute("SELECT ID, FandomName, FandomPseud FROM WorkFandoms").fetchall(), columns=['ID', 'FandomName', 'FandomPseud'])
    data['FandomName'] = data['FandomName'].map(format_tag)
    mapper = db.get_data("OriginalFandomName", "FandomName").set_index("OriginalFandomName")
    missing_data = ~data['FandomName'].isin(mapper.index)
    
    def fix_missing(s):
        if s[-9:]==' - Fandom':
            return s[:-9]
        return s
    data.loc[missing_data, 'FandomName'] = data.loc[missing_data, 'FandomName'].map(fix_missing)
    
    missing_data = ~data['FandomName'].isin(mapper.index)
    missing_data = data[missing_data]['FandomName'].value_counts().index
    missing_data = pd.DataFrame({'OriginalFandomName': missing_data, 'FandomName': [None]*len(missing_data)}).set_index('OriginalFandomName')
    mapper = pd.concat([mapper, missing_data])
    
        
    rows = [(row['ID'], mapper['FandomName'][row['FandomName']], row['FandomPseud']) for iid, row in data.iterrows()]
    if len(db.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='OriginalWorkFandoms'").fetchall()) == 0:
        db.cur.execute("CREATE TABLE OriginalWorkFandoms AS SELECT * FROM Fandoms;")
    db.cur.execute("DROP TABLE WorkFandoms;")
    db.cur.execute("CREATE TABLE WorkFandoms(ID INTEGER, FandomName TEXT, FandomPseud TEXT, FOREIGN KEY(ID) REFERENCES Works(ID), FOREIGN KEY(FandomName) REFERENCES Fandoms(FandomName));")
    db.cur.executemany("INSERT INTO WorkFandoms VALUES(?, ?, ?)", rows)
    db.con.commit()
    
def downsample(data, x='Hits', y='Kudos', N=1000):
    sorted_data = data.sort_values(by=x)
    downsample = 1000
    new_x = np.zeros(len(data)//downsample)
    new_y = np.zeros(len(data)//downsample)
    for i in range(len(data)//downsample):
        new_x[i] = np.mean(sorted_data[x].values[downsample*i:downsample*(i+1)])
        new_y[i] = np.mean(sorted_data[y].values[downsample*i:downsample*(i+1)])
    return new_x, new_y

def rolling_mean(x, N=5): # N should be odd, to get the right edges behavior below
    if len(x)<=5:
        return x
    mean = pd.Series(x).rolling(window=N).mean().iloc[N-1:].values
    for i in range(N//2, 0, -1):
        mean = np.concatenate([[np.mean(x[:2*i-1])], mean, [np.mean(x[-2*i+1:])]])
    if len(x)<5:
        return mean[(5-len(x))//2:-(5-len(x)-1)//2]
    return mean
