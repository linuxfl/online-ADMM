CXX = mpic++
INC = -I. -I../include
CFLAGS = -std=c++0x -fPIC -fvisibility=hidden -O2 -Wall -Wno-deprecated
LFLAGS = 
LIBDIR = ../lib
LIBS = 
BIN = ../bin
all:
	$(CXX) -o $(BIN)/admm admm.cpp train.cpp str_util.cpp conf_util.cpp args.cpp $(INC) $(CFLAGS) $(LFLAGS) $(addprefix -L,$(LIBDIR)) $(addprefix -l,$(LIBS))
	$(CXX) -o $(BIN)/predict predict.cpp str_util.cpp conf_util.cpp $(INC) $(CFLAGS) $(LFLAGS) $(addprefix -L,$(LIBDIR)) $(addprefix -l,$(LIBS))

clean:
	rm -rf *.o *.dSYM $(BIN)/admm
