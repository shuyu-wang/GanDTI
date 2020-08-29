import numpy as np
import os
from rdkit import Chem
from collections import defaultdict
import pickle


def createAtoms(mol):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atomDict[a] for a in atoms]
    
    return np.array(atoms)


def createBondDict(mol):
    ijbondDict = defaultdict(lambda:[])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bondDict[str(b.GetBondType())]
        ijbondDict[i].append((j, bond))
        ijbondDict[j].append((i, bond))
    return ijbondDict


def getFingerprints(atoms, bondDict, radius):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprintDict[a] for a in atoms]

    else:
        nodes = atoms
        ijedgeDict = ijbondDict

        for _ in range(radius):
            fingerprints = []
            for i, j_edge in ijedgeDict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprintDict[fingerprint])
            nodes = fingerprints

        ijedgeDict2 = defaultdict(lambda: [])
        for i, j_edge in edgeDict.items():
            for j, edge in j_edge:
                bothSide = tuple(sorted((nodes[i], nodes[j])))
                edge = edgeDict[(bothSide, edge)]    
                ijedgeDict2[i].append((j, edge))
        ijedgeDict = ijedgeDict2
    return np.array(fingerprints)

def createAdjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def splitSeqeunce(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [wordDict[sequence[i:i+ngram]]
            for i in range(len(sequence)-ngram + 1)]
    words = np.array(words)
    return words


Data = 'DUDE_train_all.txt'

with open(Data, 'r') as f:
    dataList = f.read().strip().split('\n')

dataList = [d for d in dataList if '.' not in d.strip().split()[0]]


atomDict = defaultdict(lambda: len(atomDict))
fingerprintDict = defaultdict(lambda: len(fingerprintDict))
bondDict = defaultdict(lambda: len(bondDict))
edgeDict = defaultdict(lambda: len(edgeDict))
wordDict = defaultdict(lambda: len(wordDict))

Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []
for number, data in enumerate(dataList):
    smile, sequence, interaction = data.strip().split()
    Smiles += smile +'\n'

    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
    atoms = createAtoms(mol)
    ijbondDict = createBondDict(mol)
    
    fingerprints = getFingerprints(atoms, ijbondDict, radius = 2)
    compounds.append(fingerprints)

    adjacency = createAdjacency(mol)
    adjacencies.append(adjacency)

    words = splitSeqeunce(sequence, ngram = 3)
    #sequence = [wordDict[sequence]]
    #words = np.array(sequence)
    proteins.append(words)

    interactions.append(np.array([float(interaction)]))



np.save('compounds', compounds)
np.save('adjacencies', adjacencies)
np.save('proteins', proteins)
np.save('interactions', interactions)

with open('fingerprint.pickle', 'wb') as f:
    pickle.dump(dict(fingerprintDict), f)

with open('wordDict.pickle', 'wb') as f:
    pickle.dump(dict(wordDict), f)








