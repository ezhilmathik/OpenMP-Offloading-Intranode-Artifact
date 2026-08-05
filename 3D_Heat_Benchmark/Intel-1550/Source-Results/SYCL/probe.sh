#!/bin/bash -l
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=probe-%j.out
module load slurm_setup intel-toolkit
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

echo "############ COMPOSITE (device = whole card) ############"
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
ZE_AFFINITY_MASK=0,1 ./probe_p2p

echo; echo "############ FLAT (device = one stack) ############"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
ZE_AFFINITY_MASK=0,2 ./probe_p2p      # one stack from card 0, one from card 1
