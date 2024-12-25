import sys
import pandas as pd
import numpy as np
import time


class PageRanker:
  def __init__(self, dataset, d, useEdgeWeights):
    self.time = time.time()
    dataset[0] = dataset[0].str.replace('"', '').str.strip()
    dataset[2] = dataset[2].str.replace('"', '').str.strip()
    self.d = d
    self.dataset = dataset
    self.uniqs = pd.unique(pd.concat([self.dataset[2],(self.dataset[0])]))
    self.uniqs.sort()
    self.pageRanks = np.full(len(self.uniqs), 1.0/len(self.uniqs))
    self.useEdgeWeights = useEdgeWeights


    self.adjacencies = self.makeAdjacencies()
    matrixTime = time.time()
    print("Time taken to build adjacency matrix:", matrixTime-self.time, "seconds")
    (iterations, results) = self.rankPages()
    rankTime = time.time()
    print("Time taken to rank pages:", rankTime-matrixTime, "seconds")
    print("Iterations of PageRank to converge:", iterations)
    print("________________")
    for (index, row) in results.iterrows():
      print(index+1, row["Name"], "with PageRank", row["Rank"])

  def makeAdjacencies(self):
    adMatrix = np.zeros((len(self.uniqs), len(self.uniqs)))
    for (index, row) in self.dataset.iterrows():
      rowIndex = np.nonzero(self.uniqs == row[2])[0][0]
      colIndex = np.nonzero(self.uniqs == row[0])[0][0]
      if self.useEdgeWeights:
        adMatrix[rowIndex, colIndex] += (row[1]-row[3])
      else:
         adMatrix[rowIndex, colIndex] += 1
    return adMatrix

  def getSumPart(self, j, oldRanks):
    return (oldRanks[j]/(self.adjacencies[j].sum()))
  def calculateNewRank(self, i, oldRanks):
    part1 = (1.0/len(self.uniqs)) * (1-self.d)
    leadInNodes = np.nonzero(self.adjacencies[:, i] > 0)[0]
    appliedNodes = np.array([self.getSumPart(k, oldRanks) for k in leadInNodes])
    part2 = self.d * (appliedNodes.sum())
    return part1 + part2
  def rankPages(self):
    oldRanks = self.pageRanks
    diff = 100000
    count = 0
    while diff > .001:
      count += 1
      newRanks = np.array([self.calculateNewRank(i, oldRanks) for i in range(len(self.uniqs))])
      diff = sum(abs(oldRanks-newRanks))
      oldRanks = newRanks
    results = pd.DataFrame(np.array([self.uniqs, newRanks]).transpose(), columns = ["Name", "Rank"]).sort_values("Rank", ascending=False).reset_index()
    return (count, results)

def main():
  df = pd.read_csv(sys.argv[1], header = None)
  myRanker = PageRanker(df, float(sys.argv[2]), bool(int(sys.argv[3])))

if __name__ == "__main__":
  main()


