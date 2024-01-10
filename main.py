# Dec 14, 2023. Repository was created to keep track of codings for DOE project.

from Archive.MasterSubproblemGene import TS_SPModel

if __name__ == '__main__':
    space = TS_SPModel()
    space.BuildMasterProb()
    X = {(i, j): 0 for i in range(0, 2) for j in range(1, 4)}
    space.BuildSubProb(X, 3)
