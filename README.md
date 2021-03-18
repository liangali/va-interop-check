# va-interop-check

### step 1

download internal dnnl-dldt package, below is an example

```bash
wget https://gfx-assets.igk.intel.com/artifactory/gsae-solutions-igk/dnnl/dnnl-ats/dldt-dnnl-3273/archive.zip
mv archive.zip dldt-dnnl-3273.zip
mkdir 3273
unzip dldt-dnnl-3273.zip -d ./3273
```
### step 2

get openvino header files

```bash
git clone https://github.com/openvinotoolkit/openvino.git
# depends on which dnnl-dldt package is used, need check out different openvino branch, 
# for example dldt-dnnl-3273 needs to use releases/2021/2
git checkout releases/2021/2
```
### step 3

build va-interop-check app

*** We must set correct DLDT_DIR and DLDT_INCLUDE_DIR env variables before run CMake ***
```bash
export DLDT_DIR=/home/fresh/data/work/va_solution_innersource/dldt/3273/Linux
export DLDT_INCLUDE_DIR=/home/fresh/data/work/va_solution_innersource/source/openvino/inference-engine
export LD_LIBRARY_PATH=${DLDT_DIR}/lib:${DLDT_DIR}/opencv/lib:${DLDT_DIR}/tbb/lib
```
we can copy above lines into "env.sh" and run "source env.sh"

```bash
cd va-interop-check
source env.sh
mkdir build
cd build
cmake ..
make
```

### step 4

run va-interop-check app

```bash
cd build
./interop 
```

this app has build-in test clips, if you find below messages in terminal log, it indicates the test passed

```
INFO: decode Y-plane matches with reference
INFO: decode UV-plane matches with reference
......
INFO: Resnet50 inference output: batchIndex = 0, classID = 240, confidence = 0.779041
INFO: Resnet50 inference output: batchIndex = 1, classID = 240, confidence = 0.779041
```
