
# Simulate data
```bash
sbatch --job-name sim-data --time 100:00:00 --mem 80gb --partition jro0014_amd \
    --output /scratch/phyletica/theta-cnn/%x%j.out \
    --wrap "./sim.py S20_N5000_L1e5_M1e-4_R1e-8.yaml /scratch/phyletica/theta-cnn"
```

# Run CNN
```bash
sbatch --job-name cnn --time 48:00:00 --mem 40gb --partition jro0014_amd \
    --output /scratch/phyletica/theta-cnn/%x%j.out \
    --wrap "./cnn.py /scratch/phyletica/theta-cnn/S20_N1000_L1e4_M1e-4_R1e-8.npz /scratch/phyletica/theta-cnn/"
```