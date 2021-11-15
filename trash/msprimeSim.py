import msprime
import numpy as np
from sklearn.neighbors import NearestNeighbors

def simSeqData(seed, nDataSets, nSamples, sequenceLength, mutationRate):
    # Simulate data with msprime
    lowerPopSize = 1e3
    upperPopSize = 1e5
    recombRate = 1e-8
    # Create empty numpy arrays to hold data
    popSizeArray = np.empty(nDataSets, dtype=np.int64)
    varCharMatrixArray = np.empty(nDataSets, dtype=object)
    invarCharMatrixArray = np.zeros((nDataSets, sequenceLength, 
            nSamples * 2), dtype=np.int8)
    for rep in range(nDataSets):
        # Draw random population size 
        popSize = np.random.randint(lowerPopSize, upperPopSize)
        popSizeArray[rep] = popSize 
        # Create tree sequences with msprime 
        ts = msprime.sim_ancestry(samples=nSamples, 
                recombination_rate=recombRate, sequence_length=sequenceLength, 
                population_size=popSize, random_seed=seed)
        mutated_ts = msprime.sim_mutations(ts, rate=mutationRate, 
                random_seed=seed, model=msprime.BinaryMutationModel())
        # Get data matrix with only variable sites
        varCharMatrixArray[rep] = mutated_ts.genotype_matrix()
        # Get complete character matrix with invariant sites
        for i in mutated_ts.variants():
            site = int(i.site.position)
            invarCharMatrixArray[rep][site,:] = i.genotypes
    return (popSizeArray, varCharMatrixArray, invarCharMatrixArray)

def similarity_ranks(matrices):
    ranks = np.empty(matrices.shape[0], dtype=object)
    for i, mat in enumerate(matrices):
        mat = mat.transpose() 
        nbrs = NearestNeighbors(n_neighbors=mat.shape[0], metric="manhattan").fit(mat)
        distances, ixs = nbrs.kneighbors(mat)
        smallest = np.argmin(distances.sum(axis=1))
        ranks[i] = ixs[smallest]
    return ranks 

def sort_matrices(ranks, matrices):
    for i in range(matrices.shape[0]):
        matrices[i] = matrices[i][:,ranks[i]]

def padMatrices(matrices):
    maxLen = max([i.shape[0] for i in matrices]) 
    paddedMatrixArray = np.zeros((matrices.shape[0], maxLen, 
            matrices[0].shape[1]), dtype=np.int8)
    for i in range(matrices.shape[0]):
        for j in range(matrices[i].shape[0]):
            paddedMatrixArray[i][j] = matrices[i][j]
    return paddedMatrixArray

#-------------------------------------------------------------------------------
# Simulate data with msprime
popSizeArray, varCharMatrixArray, invarCharMatrixArray = simSeqData(
    seed=1234, 
    nDataSets=1, 
    nSamples=2, 
    sequenceLength=6, 
    mutationRate=1e-4)

# Sort matrices by similarity
ranks = similarity_ranks(varCharMatrixArray)
sort_matrices(ranks, varCharMatrixArray)
sort_matrices(ranks, invarCharMatrixArray)

# Pad variant matices with zeros
padVarCharMatrixArray = padMatrices(varCharMatrixArray)

# Compress and save data 
np.savez_compressed(
    "simulated-data.npz",
    populationSizeArray=popSizeArray,
    variantMatrixArray=padVarCharMatrixArray,
    invariantMatrixArray=invarCharMatrixArray)
