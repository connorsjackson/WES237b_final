CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

solution: ../../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a: 
	cd ../../helper_lib; make

run: solution
	./solution Dataset/0/input0.raw Dataset/0/input1.raw Dataset/0/output0.raw Dataset/0/output1.raw Dataset/0/expected0.raw Dataset/0/expected1.raw
	./solution Dataset/1/input0.raw Dataset/1/input1.raw Dataset/1/output0.raw Dataset/1/output1.raw Dataset/1/expected0.raw Dataset/1/expected1.raw
	./solution Dataset/2/input0.raw Dataset/2/input1.raw Dataset/2/output0.raw Dataset/2/output1.raw Dataset/2/expected0.raw Dataset/2/expected1.raw
	./solution Dataset/3/input0.raw Dataset/3/input1.raw Dataset/3/output0.raw Dataset/3/output1.raw Dataset/3/expected0.raw Dataset/3/expected1.raw
	./solution Dataset/4/input0.raw Dataset/4/input1.raw Dataset/4/output0.raw Dataset/4/output1.raw Dataset/4/expected0.raw Dataset/4/expected1.raw
	./solution Dataset/5/input0.raw Dataset/5/input1.raw Dataset/5/output0.raw Dataset/5/output1.raw Dataset/5/expected0.raw Dataset/5/expected1.raw
	./solution Dataset/6/input0.raw Dataset/6/input1.raw Dataset/6/output0.raw Dataset/6/output1.raw Dataset/6/expected0.raw Dataset/6/expected1.raw
	./solution Dataset/7/input0.raw Dataset/7/input1.raw Dataset/7/output0.raw Dataset/7/output1.raw Dataset/7/expected0.raw Dataset/7/expected1.raw
	
run1: solution
	./solution Dataset/2/input0.raw Dataset/2/input1.raw Dataset/2/output0.raw Dataset/2/output1.raw Dataset/2/expected0.raw Dataset/2/expected1.raw

clean: 
	rm -f solution
	rm ../../helper_lib/helper_lib.a
	cd ../../helper_lib; make clean	
