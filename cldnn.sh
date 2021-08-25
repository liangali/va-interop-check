export DLDT_DNNL=off
export BATCH_SHARE=on

export OV_SOURCE_DIR=/home/fresh/data/work/dg2_va_solution/source/openvino.private-fork-ww35

export DLDT_INCLUDE_DIR=${OV_SOURCE_DIR}/inference-engine
export DLDT_DIR=${OV_SOURCE_DIR}/bin/intel64/Debug
export LD_LIBRARY_PATH=${DLDT_DIR}/lib:${DLDT_DIR}/opencv/lib:${DLDT_DIR}/tbb/lib

export PYTHONPATH=${DLDT_DIR}/lib/python_api/python3.8

# export NEOReadDebugKeys=1

# export NodeOrdinal=0

# export EnableBlitterForEnqueueOperations=1

# export ForcePreemptionMode=2
