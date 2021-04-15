from timeit import default_timer as timer
import datetime
import msprime
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def simSeqData(seed, nDataSets, nSamples, sequenceLength, mutationRate):
    # Simulate data with msprime
    lowerPopSize = 1e3
    upperPopSize = 1e5
    recombRate = 1e-8

    print("Simulating datasets:")
    print(f"\tseed: {seed}")
    print(f"\tnumber of data sets: {nDataSets}")
    print(f"\tsequence length: {sequenceLength}")
    print(f"\tmutationRate: {mutationRate}")
    print(f"\tpopulation size range: {lowerPopSize}-{upperPopSize}")
    print(f"\trecombination rate: {recombRate}")

    # Create empty numpy arrays to hold data
    populationSizeArray = np.empty(nDataSets, dtype=np.int64)
    variantCharacterMatrixArray = np.empty(nDataSets, dtype=object)
    invariantCharacterMatrixArray = np.zeros((nDataSets, sequenceLength, nSamples * 2), dtype=np.int8)
    for rep in tqdm(range(nDataSets)):
        # Draw random population size 
        populationSize = np.random.randint(lowerPopSize, upperPopSize)
        populationSizeArray[rep] = populationSize 

        # Create tree sequences with msprime 
        ts = msprime.sim_ancestry(samples=nSamples, recombination_rate=recombRate, 
                sequence_length=sequenceLength, population_size=populationSize, 
                random_seed=seed)
        mutated_ts = msprime.sim_mutations(ts, rate=mutationRate, random_seed=seed, 
                    model=msprime.BinaryMutationModel())

        # Get data matrix with only variable sites
        variantCharacterMatrixArray[rep] = mutated_ts.genotype_matrix()
        
        # Get complete character matrix with invariant sites
        for i in mutated_ts.variants():
            site = int(i.site.position)
            invariantCharacterMatrixArray[rep][site,:] = i.genotypes
    return (populationSizeArray, variantCharacterMatrixArray, invariantCharacterMatrixArray)

def sort_min_diff(amat):
    """
    This function takes in a SNP matrix with individuals on rows and returns 
    the same matrix with individuals sorted by genetic similarity.
    This problem is NP, so here we use a nearest neighbors approximation. It"s 
    not perfect, but it"s fast and generally performs ok. This assumes your 
    input matrix is a numpy array.

    Taken from Flagel et al. 2019, doi:10.1093/molbev/msy224 
    """ 
    # TODO: Rewrite to sort on columns
    # TODO: Make len(amat) more efficient
    # print(len(amat))
    # print(amat.shape)
    mb = NearestNeighbors(len(amat), metric="manhattan").fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

def padMatrices(matrices):
    maxLen = 0
    nSamples = matrices[0].shape[1] 
    for i in matrices:
        if i.shape[0] > maxLen:
            maxLen = i.shape[0]
    paddedMatrixArray = np.zeros((matrices.shape[0], maxLen, nSamples), dtype=np.int8)
    for i in range(matrices.shape[0]):
        for j in range(matrices[i].shape[0]):
            paddedMatrixArray[i][j] = matrices[i][j]
    return paddedMatrixArray

#-------------------------------------------------------------------------------
# Simulate data with msprime
start1 = timer()
populationSizeArray, variantCharacterMatrixArray, invariantCharacterMatrixArray = simSeqData(
    seed=1234, 
    nDataSets=5000, 
    nSamples=20, 
    # sequenceLength=50_000, 
    sequenceLength=10_000, 
    mutationRate=1e-7)
end1 = timer()
elapsed1 = str(datetime.timedelta(seconds=round(end1 - start1)))
print(f"Elapsed time for data simulation: {elapsed1}")

#TODO: Sort matrices by similarity

# Pad variant matices with zeros
paddedVariantCharacterMatrixArray = padMatrices(variantCharacterMatrixArray)

# Split data sets into testing and training
outputFile = f"simulated-data-{invariantCharacterMatrixArray[0].shape[0]}.npz"
print(f"Saving compressed data to {outputFile}")
start2 = timer()
np.savez_compressed(
    outputFile,
    populationSizeArray=populationSizeArray,
    variantMatrixArray=paddedVariantCharacterMatrixArray,
    invariantMatrixArray=invariantCharacterMatrixArray)
end2 = timer()
elapsed2 = str(datetime.timedelta(seconds=round(end1 - start1)))
print(f"Elapsed time for data compression and output: {elapsed2}")
