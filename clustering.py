import pandas as pd
import numpy as np
import json
import matplotlib as plt
import seaborn as sns
from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import sys



def cleanDataset(link):
  df = pd.read_csv(link, header = None)
  df = df.drop(columns=df.columns[df.iloc[0] == '0'])
  return (df.iloc[1:,:]).reset_index().drop("index", axis = 1)
def cleanDatasetAccidents(link):
  df = pd.read_csv(link, header = None, usecols = range(5))
  df = df.drop(columns=df.columns[df.iloc[0] == '0'])
  return (df.iloc[1:,:]).reset_index().drop("index", axis = 1)

class Clusterer(ABC):
  @abstractmethod
  def makeClusters(self):
    raise NotImplementedError("Not yet implemented")

class ClusterTree:
  def __init__(self, points, left, right, height, isLeaf):
    self.points = points
    self.left = left
    self.right = right
    self.height = height
    self.isLeaf  = isLeaf

  def __repr__(self):
    return self.asString()
  def asString(self):
    return "Height: " + str(self.height) + ("\n   Left: " + self.left.asString() if self.left != None else "") + (", Right: " + self.right.asString() if self.right != None else "")
  def makeDictionary(self):
    returnDict = {}
    returnDict["type"] = "leaf" if self.isLeaf else "root"
    returnDict["height"] = self.height

    childList = [self.left.makeDictionary(), self.right.makeDictionary()] if not self.isLeaf else []
    if not self.isLeaf:
      returnDict["nodes"] = childList
    else:
      (index, row) = self.points[0]
      returnDict["data"] = row.to_string()
    return returnDict
  def makeList(self, threshold):
    if self.height < threshold:
      return [self]
    else:
      return self.left.makeList(threshold) + self.right.makeList(threshold)



class HierarchicalClusterer(Clusterer):
  def __init__(self, dataset: pd.DataFrame, threshold: int):
    self.clusters = []
    self.dataset = dataset
    self.distances = None
    self.flatDistances = None
    self.pointsToClusters = {}
    self.threshold = threshold
  def findDistances(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists

  def calculateDistances(self):
    self.distances = np.array([np.array(xi) for xi in squareform(pdist(self.dataset, 'euclidean'))])
    self.distances[np.tril_indices(self.distances.shape[0])] = np.inf
    self.flatDistances = pdist(self.dataset, "euclidean")
  def showPlot(self):
    self.calculateDistances()
    links = linkage(self.flatDistances, "single")
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(links)
    plt.show()

  def makeInitialClusters(self):
    for index, row in self.dataset.iterrows():
      newCluster = ClusterTree([(index,row)], None, None, 0, True)
      self.clusters.append(newCluster)
      self.pointsToClusters[index] = newCluster

  def minDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.min()

  def maxDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.max()

  def avgDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.mean()

  def SSE(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = ((df - row) ** 2).sum(axis=1)
    return dists.sum()

  def findSmallestDistance(self):
    (smallRow, smallCol) = np.unravel_index(np.argmin(self.distances, axis=None), self.distances.shape)
    smallDist = self.distances[smallRow][smallCol]
    self.distances[smallRow][smallCol] = np.inf
    return (smallRow, smallCol, smallDist)

  def mergeClusters(self, c1, c2, height):
    newCluster = ClusterTree(c1.points + c2.points, c1, c2, height, False)
    for (i, row) in c1.points:
      self.pointsToClusters[i] = newCluster
    for (i, row) in c2.points:
      self.pointsToClusters[i] = newCluster
    for (i, row) in newCluster.points:
      for (i2, row2) in newCluster.points: #sorry Lana :/ optimize later if not feeling lazy
        if i < i2:
          self.distances[i][i2] = np.inf
    return newCluster
  def printMetrics(self, c):
    thisCluster = self.dataset.iloc[[i for (i, j) in c.points]]
    thisCentroid = thisCluster.mean()
    print("Cluster #" + str(self.clusters.index(c)) + ":")
    members = [i for (i, j) in c.points]
    print("Members:", members)
    print("Centroid:", thisCentroid.tolist()[:-1])
    minDist = self.minDist(thisCentroid, thisCluster)
    print("Min distance:", minDist)
    maxDist = self.maxDist(thisCentroid, thisCluster)
    print("Max distance:", maxDist)
    avgDist = self.avgDist(thisCentroid, thisCluster)
    print("Average distance:", avgDist)
    SSE = self.SSE(thisCentroid, thisCluster)
    print("Sum Squared Errors:", SSE)

  def makeClusters(self):
    self.calculateDistances()
    self.makeInitialClusters()
    while len(self.clusters) > 1:
      (x, y, height) = self.findSmallestDistance()
      (firstCluster, secondCluster) = (self.pointsToClusters[x], self.pointsToClusters[y])
      newCluster = self.mergeClusters(firstCluster, secondCluster, height)
      self.clusters.remove(firstCluster)
      self.clusters.remove(secondCluster)
      self.clusters.append(newCluster)
    if self.threshold != 0:
      self.clusters = self.clusters[0].makeList(self.threshold)
      print("Num clusters:", len(self.clusters))
      for c in self.clusters:
        self.printMetrics(c)
    return self.clusters[0].makeDictionary()

class DBScanClusterer(Clusterer):
  def __init__(self, dataset, radius, minPoints):
    self.clusters = []
    self.dataset = dataset
    self.distances = None
    self.pointsToClusters = {}
    for (index, row) in self.dataset.iterrows():
      self.pointsToClusters[index] = None
    self.neighbors = None
    self.radius = radius
    self.minPoints = minPoints
    self.noisePoints = []

  def calculateDistances(self):
    self.distances = pd.DataFrame(np.array([np.array(xi) for xi in squareform(pdist(self.dataset, 'euclidean'))]))
  def minDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.min()

  def maxDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.max()

  def avgDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.mean()

  def SSE(self, row: pd.Series, df: pd.DataFrame) -> int:
    dists = ((df - row) ** 2).sum(axis=1)
    return dists.sum()

  def findNeighbors(self):
    self.neighbors = self.distances.le(self.radius).apply(lambda x : x.index[x].tolist(), axis = 1)
    self.distances['counts'] = self.distances[self.distances < self.radius].count()-1
    self.distances['isCore'] = self.distances['counts'] >= self.minPoints
    self.distances = self.distances.drop(['counts'], axis = 1)

  def densityConnected(self, point, cluster):
    for p in self.neighbors[point]:
      currentVal = self.pointsToClusters[p]
      self.pointsToClusters[p] = cluster
      if self.distances.at[p, 'isCore'] == True and currentVal != cluster:
        self.densityConnected(p, cluster)
  def printMetrics(self):
    print("Clusters:")
    for c in self.clusters:
      self.printClusterMetrics(c)
    print("Outliers:", self.noisePoints)
    print("Percentage of outliers:", str(100*len(self.noisePoints)/len(self.dataset.index))+"%")

  def printClusterMetrics(self, c):

    thisCluster = self.dataset.iloc[c]
    thisCentroid = thisCluster.mean()
    print("Cluster #" + str(self.clusters.index(c)) + ":")
    members = c
    print("Members:", members)
    print("Centroid:", thisCentroid.tolist()[:-1])
    minDist = self.minDist(thisCentroid, thisCluster)
    print("Min distance:", minDist)
    maxDist = self.maxDist(thisCentroid, thisCluster)
    print("Max distance:", maxDist)
    avgDist = self.avgDist(thisCentroid, thisCluster)
    print("Average distance:", avgDist)
    SSE = self.SSE(thisCentroid, thisCluster)
    print("Sum Squared Errors:", SSE)


  def makeClusters(self):
    self.calculateDistances()
    self.findNeighbors()
    currentCluster = 0
    indices = self.distances.index[self.distances['isCore']].tolist()
    for i in indices:
      if self.pointsToClusters[i] == None:
        self.pointsToClusters[i] = currentCluster
        self.densityConnected(i, currentCluster)
        currentCluster += 1
    for i in set(self.pointsToClusters.values()):
      self.clusters.append([])
    for i in self.pointsToClusters.keys():
      if self.pointsToClusters[i] != None:
        self.clusters[self.pointsToClusters[i]].append(i)
      else:
        self.noisePoints.append(i)
    if self.clusters[-1] == []:
      self.clusters.pop(-1)
    self.printMetrics()
    return (self.clusters, self.noisePoints)
    
class KMeansClusterer(Clusterer):
  def __init__(self, k, dataset):
    self.k = k
    self.dataset = dataset
    self.clusters = []
    self.centroids = None #Random k selection of
  def randomCentroids(self):
    self.centroids = self.dataset.sample(n=self.k).reset_index(drop = True)
  def distances(self, row: pd.Series, df: pd.DataFrame) -> int:
    "computes distances of each point in df and returns index of closest centroid"
    df = df.drop(['centroid', 'new centroid'], axis=1, errors='ignore')
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.idxmin()
  def minDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    df = df.drop(['centroid', 'new centroid'], axis=1, errors='ignore')
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.min()
  def maxDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    df = df.drop(['centroid', 'new centroid'], axis=1, errors='ignore')
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.max()
  def avgDist(self, row: pd.Series, df: pd.DataFrame) -> int:
    df = df.drop(['centroid', 'new centroid'], axis=1, errors='ignore')
    dists = np.sqrt(((df - row) ** 2).sum(axis=1))
    return dists.mean()
  def SSE(self, row: pd.Series, df: pd.DataFrame) -> int:
    df = df.drop(['centroid', 'new centroid'], axis=1, errors='ignore')
    dists = ((df - row) ** 2).sum(axis=1)
    return dists.sum()

  def calculateCentroids(self):
    self.dataset['centroid'] = self.dataset.apply(lambda row: self.distances(row, self.centroids), axis=1)
  def recomputeCentroids(self):
    self.centroids = self.dataset.groupby("centroid").mean()
    self.dataset['new centroid'] = self.dataset.apply(lambda row: self.distances(row, self.centroids), axis=1)

  def centroidChanges(self):
    n = len(self.dataset.index)

    stop = n * 0.02

    self.dataset['diff'] = (self.dataset['centroid'] != self.dataset['new centroid'])

    changes = self.dataset['diff'].sum()

    self.dataset = self.dataset.drop(['diff'], axis=1, errors='ignore')

    if changes > stop:
      return True
    return False
  def printMetrics(self):
    for i, row in self.centroids.iterrows():
      print("Cluster #"+str(i)+":")
      members = self.dataset.index[self.dataset['new centroid'] == i].tolist()
      print("Members:", members)
      print("Coordinates:", row.tolist()[:-1])
      minDist = self.minDist(row, self.dataset[(self.dataset['new centroid'] == i)])
      print("Min distance:",minDist)
      maxDist = self.maxDist(row, self.dataset[(self.dataset['new centroid'] == i)])
      print("Max distance:",maxDist)
      avgDist = self.avgDist(row, self.dataset[(self.dataset['new centroid'] == i)])
      print("Average distance:",avgDist)
      SSE = self.SSE(row, self.dataset[(self.dataset['new centroid'] == i)])
      print("Sum Squared Errors:", SSE)



  def makeClusters(self):
    self.randomCentroids()
    self.calculateCentroids()
    self.recomputeCentroids()
    counter = 0
    while self.centroidChanges():
      self.dataset["centroid"] = self.dataset["new centroid"]
      self.recomputeCentroids()
      counter += 1
    self.printMetrics()


if __name__ == "__main__":
  dataset = cleanDataset(sys.argv[2])
  type = sys.argv[3]
  k = -1
  epsilon = -1
  minPoints = -1
  if type == "k":
    k = sys.argv[4]
    clusterer = KMeansClusterer(k, dataset)
    clusterer.makeClusters()
  if type == "d":
    epsilon = sys.argv[4]
    minPoints = sys.argv[5]
    clusterer = DBScanClusterer(dataset, epsilon, minPoints)
    (clusters, noise) = clusterer.makeClusters()
  if type == "h":
    clusterer = HierarchicalClusterer(dataset, 0)
    clusterer.makeClusters()
  
  
  
