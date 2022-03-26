#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=2:00:00"
#PJM --llio cn-cache-size=1Gi
#PJM --llio sio-read-cache=on
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -x PJM_LLIO_GFSCACHE=/vol0003
#PJM --mpi "proc=16"
#PJM --mpi "max-proc-per-node=16"
#PJM -s

source /vol0004/apps/oss/spack/share/spack/setup-env.sh
export MODULEPATH=/vol0004/apps/arm-lib/modulefiles:$MODULEPATH
export MODULEPATH=/vol0004/apps/oss/spack/share/spack/modules/linux-rhel8-a64fx:$MODULEPATH

spack load /vgsk3aw
make
mpiexec -std out/output.txt -n 16 ./lorasp
