#!/usr/bin/env python3

import msprime
import numpy as np
import pickle
import fire
import yaml
import os.path as p

def simulate(configFile, outdir):
    config = yaml.load(open(configFile, "r"), Loader=yaml.FullLoader) 
    seed = config["seed"] 
    nDataSets = config["nDataSets"] 
    nSamples = config["nSamples"]
    seqLength = int(float(config["seqLength"]))
    recombRate = float(config["recombRate"])
    mutRate = float(config["mutRate"]) 
    lowerSize = float(config["lowerSize"])
    upperSize = float(config["upperSize"])

    popSizeArray = np.empty(nDataSets, dtype=np.uint32)
    positionArray = np.empty(nDataSets, dtype=object)
    varCharMatrices = np.empty(nDataSets, dtype=object)
    invarCharMatrices = np.zeros((nDataSets, seqLength, nSamples * 2), dtype=np.uint8)
    
    for rep in range(nDataSets):
        print(f"Simulating dataset {rep}", flush=True)
        popSize = np.random.randint(lowerSize, upperSize)
        popSizeArray[rep] = popSize 
        ts = msprime.sim_ancestry(samples=nSamples, recombination_rate=recombRate, 
                sequence_length=seqLength, population_size=popSize, random_seed=seed)
        mts = msprime.sim_mutations(ts, rate=mutRate, random_seed=seed, 
                model=msprime.BinaryMutationModel())
        positionArray[rep] = mts.tables.sites.position.astype(np.int32) 
        varCharMatrices[rep] = mts.genotype_matrix()
        for i in mts.variants():
            site = int(i.site.position)
            invarCharMatrices[rep][site,:] = i.genotypes
    
    metaData = dict(
        nDataSets=nDataSets,
        nSamples=nSamples,
        seqLength=seqLength,
        recombRate=recombRate,
        mutationRate=mutRate,
        lowerPopSize=lowerSize,
        upperPopSize=upperSize,
        seed=seed)
    data = dict(
        metaData=pickle.dumps(metaData),
        popSizes=popSizeArray,
        positions=positionArray,
        varChars=varCharMatrices,
        invarChars=invarCharMatrices)
    outfile = f"{p.splitext(p.basename(configFile))[0]}.npz"
    outpath = p.join(outdir, outfile)
    np.savez_compressed(outpath, **data)
    print("Simulation complete")

if __name__ == "__main__":
    fire.Fire(simulate)