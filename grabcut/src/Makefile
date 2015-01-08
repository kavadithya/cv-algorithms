CXX=g++
CXXFLAGS=-std=c++11 -Wall -O3 -g -MMD -MP
OPENCV=`pkg-config opencv --cflags --libs`
LIBS=$(OPENCV)
TARGET=main

all: $(TARGET)

$(TARGET): main.cpp grabcut.cpp include/gmm/gmm.cpp
		$(CXX) $(CXXFLAGS) $(LIBS) $+ -o $@

clean:
		$(RM) -r *.o *.d *.dSYM *.txt $(TARGET)

.PHONY: all clean
