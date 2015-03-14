CFLAGS = -O4 -I/usr/local/cuda-5.5/include -L/usr/local/cuda/lib64 -lcudart

all: executable

executable: kernels.o
	mpiCC $(CFLAGS) main.cc network.cc kernels.o -o executable

kernels.o:
	nvcc -O4 -c kernels.cu

clean:
	rm kernels.o

