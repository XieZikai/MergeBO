#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job
#SBATCH --export=ALL
# Define job name
#SBATCH -J geom_optvasp
# Define a standard output file. When the job is running, %u will be replaced by user name,
# %N will be replaced by the name of the node that runs the batch script, and %j will be replaced by job id number.
#SBATCH -o vasp.%u.%N.%j.out
#SBATCH -e vasp.%u.%N.%j.err
#SBATCH -N 1 -p dzacexclu01
#SBATCH -n 2
# Specify time limit in format a-bb:cc:dd, where a is days, b is hours, c is minutes, and d is seconds.
#SBATCH -t 10-00:00:00
#SBATCH -a 1-20
ulimit -s unlimited

#
# Should not need to edit below this line
#
echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

echo Executable file:                              
echo MPI parallel job.                                  
echo -------------  
echo Job output begins                                           
echo -----------------                                           
echo

hostname

echo "Print the following environmetal variables:"
echo "Job name                     : $SLURM_JOB_NAME"
echo "Job ID                       : $SLURM_JOB_ID"
echo "Job user                     : $SLURM_JOB_USER"
echo "Job array index              : $SLURM_ARRAY_TASK_ID"
echo "Submit directory             : $SLURM_SUBMIT_DIR"
echo "Temporary directory          : $TMPDIR"
echo "Submit host                  : $SLURM_SUBMIT_HOST"
echo "Queue/Partition name         : $SLURM_JOB_PARTITION"
echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Hostname of 1st node         : $HOSTNAME"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"

echo "Running parallel job:"

module purge
conda init
source activate merge 

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
i=$SLURM_ARRAY_TASK_ID
j=$((i-1))
WORKDIR="cp_merge_no_anchor_${i}"

mkdir -p $WORKDIR

cp cp_simulator cp_merge_no_anchor.py bo_pl.mat ex_10_40_2_10.dat output_pl.txt permutation.txt run_sdp_pl.m $WORKDIR/
cd $WORKDIR/
chmod +x cp_simulator

python cp_merge_no_anchor.py --start $j --end $i

cd ..

echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
exit $ret
