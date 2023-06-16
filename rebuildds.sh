find ./ -name "*.so" | xargs rm
# find ./ -name "build" | xargs rm
python setup.py clean

CUDA_INC_PATH=/home/baodi/workspace/cuda_12.0 \
CC=icx \
CXX=icpx \
CFLAGS=-fPIC \
CXXFLAGS=-fPIC \
DS_BUILD_DEVICE=dpcpp \
DS_BUILD_TRANSFORMER_INFERENCE=1 \
python setup.py develop
