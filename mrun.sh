#!/bin/bash
echo $1 $2 $3

PWD=/home/mfa51/service-capacity

if [ $1 = 'i' ]; then
  module load intel/16.0.3 mvapich2/2.1
  # source ~/tensorflow/bin/activate
elif [ $1 = 'r' ]; then
  FILE='load_imbalance' # 'bucket_sim'
  NTASKS=1
  echo "#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=$FILE
#SBATCH --nodes=$NTASKS              # Number of nodes you require
#SBATCH --ntasks=$NTASKS             # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=4000                   # Real memory (RAM) required (MB)
#SBATCH --time=3:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
#SBATCH --output=log/slurm.%N.%j.out
export MV2_ENABLE_AFFINITY=0
# srun --mpi=pmi2 python3 $PWD/$FILE.py
source /home/mfa51/service-capacity-virtualenv/bin/activate
python3 $PWD/$FILE.py
  " > jscript.sh
  
  # rm log/*
  sbatch jscript.sh
elif [ $1 = 'l' ]; then
  squeue -u mfa51
elif [ $1 = 'k' ]; then
  scancel --user=mfa51 # -n learning 
else
  echo "Arg did not match!"
fi
