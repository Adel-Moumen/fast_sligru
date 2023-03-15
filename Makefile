AR ?= ar
CXX ?= g++
NVCC ?= nvcc -ccbin $(CXX)
PYTHON ?= python
HDR = src/*.h

OBJ_DIR=build/obj
OBJ = \
	$(OBJ_DIR)/layer_norm_forward_gpu.o \
	$(OBJ_DIR)/layer_norm_backward_gpu.o \
	$(OBJ_DIR)/ligru_1_0_forward_gpu.o \
	$(OBJ_DIR)/ligru_1_0_backward_gpu.o \
	$(OBJ_DIR)/ligru_2_0_forward_gpu.o \
	$(OBJ_DIR)/ligru_2_0_backward_gpu.o

ifeq ($(OS),Windows_NT)
LIBHASTE := libhaste.lib
CUDA_HOME ?= $(CUDA_PATH)
AR := lib
AR_FLAGS := /nologo /out:$(LIBHASTE)
NVCC_FLAGS := -x cu -Xcompiler "/MD"
else
LIBHASTE := libhaste.a
CUDA_HOME ?= /usr/local/cuda
AR ?= ar
AR_FLAGS := -crv $(LIBHASTE)
NVCC_FLAGS := -std=c++11 -x cu -Xcompiler -fPIC
endif

LOCAL_CFLAGS := -I/usr/include/eigen3 -I$(CUDA_HOME)/include -Isrc -O3 -use_fast_math --compiler-options -fPIC
LOCAL_LDFLAGS := -L$(CUDA_HOME)/lib64 -L. -lcudart -lcublas
GPU_ARCH_FLAGS := -gencode arch=compute_60,code=compute_60 -gencode arch=compute_70,code=compute_70


all: fast_ligru clean

$(OBJ_DIR)/%.o: src/%.cu $(HDR)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c -o $@ $< $(GPU_ARCH_FLAGS) $(LOCAL_CFLAGS)

haste: $(OBJ)
	$(AR) $(AR_FLAGS) $(OBJ)

fast_ligru:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q bdist_wheel)
	@cp $(TMP)/dist/*.whl .
	@rm -rf $(TMP)

dist:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cp build/MANIFEST.in $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q sdist)
	@cp $(TMP)/dist/*.tar.gz .
	@rm -rf $(TMP)

clean:
	rm -fr *.whl
	find . \( -iname '*.o' -o -iname '*.so' -o -iname '*.a' -o -iname '*.lib' \) -delete
