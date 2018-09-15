.SUFFIXES = .cpp .cu .o

PROGRAM		= HGUNeuralNetwork

MAINSRC		= main.cpp
MAINOBJ		= ${MAINSRC:.cpp=.o}

SRCS		=	\
		HGUAutoEncoder.cpp	\
		HGULayer.cpp		\
		HGUNeuralNetwork.cpp		\
		HGURBM.cpp

ifeq ($(CUDA), ENABLE)
CUDA_SRCS = \
		HGULayer_CUDA.cu
endif	# ifeq($(CUDA), ENABLE)

INCLUDDES = \
		HGUAutoEncoder.h	\
		HGULayer.h		\
		HGUNeuralNetwork.h		\
		HGURBM.h

OBJS		= ${SRCS:.cpp=.o}
CUDA_OBJS	= ${CUDA_SRCS:.cu=.o}
LIBS		= -lm

CFLAGS		= -pthread -Wno-unused-result -O2 -std=c++11

#DFLAGS		= -g -D_DEBUG
DFLAGS		= -g 

#-D__cplusplus
CC		= g++
NVCC	= nvcc
LINKER	= g++

ifeq ($(CUDA), ENABLE)
	LINKER	= nvcc
	CFLAGS	= -pthread -Wno-unused-result -O2 -std=c++11 -D__CUDA__
endif	# ifeq($(CUDA), LINUX)


INCLUDE_PATH	= -I.
LIB_PATH= -L. -L/usr/local/cuda/lib64

all: $(PROGRAM)

#lib:	$(LIB)
#$(OBJS):	$(SRCS) $(INCLUDES)
$(MAINOBJ): $(MAINSRC)


$(PROGRAM): $(OBJS) $(CUDA_OBJS) $(MAINOBJ)
	$(LINKER) $(DFLAGS) $(OBJS) $(CUDA_OBJS) $(MAINOBJ) $(LIBS) -o $@

#$(PROGRAM_CUDA): $(OBJS) $(CUDA_OBJS) $(MAINOBJ)
#	$(NVCC) $(DFLAGS) $(OBJS) $(CUDA_OBJS) $(MAINOBJ) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(INCLUDE_PATH) -c $<

.cc.o:
	$(CC) $(CFLAGS) $(DFLAGS) $(INCLUDE_PATH) -c $<

.cu.o:
	$(NVCC) $(DFLAGS) $(INCLUDE_PATH) -c $<

HGULayer_CUDA.o: HGULayer_CUDA.cu
	$(NVCC) $(DFLAGS) $(INCLUDE_PATH) -c $<

depend:
	makedepend $(INCLUDE_PATH) *.cpp

clean:
	rm -f *.o *.*~ $(PROGRAM) $(PROGRAM_CUDA) $(LIB)
clearall:
	rm -f *.o *.*~ $(PROGRAM) $(PROGRAM_CUDA) $(LIB)

#$(PROGRAM):	$(OBJS) $(MAINOBJ)
#	$(CC) -o $@ $(CFLAGS) $(DFLAGS) $(LIB_PATH) $(LIBS) $(OBJS) $(MAINOBJ)


#$(PROGRAM):	$(LIB) $(MAINOBJ)
#	$(CC) -o $@ $(CFLAGS) $(DFLAGS) $(LIB_PATH) $(LIBS) $(OBJS) $(MAINOBJ)
#	$(CC) -o $@ $(CFLAGS) $(DFLAGS) $(LIB_PATH) $(LIBS) $(MAINOBJ) -lwindows
$(LIB): $(OBJS)
	ar -r $@ $(OBJS)

#backup:
#	tar cvf - ../include/*.h *.cpp | gzip -c > ../library.tgz
