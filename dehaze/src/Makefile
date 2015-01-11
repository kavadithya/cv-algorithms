CXX=g++
CXXFLAGS=-std=c++11 -Wall -O2 -g -MMD -MP
OPENCV=`pkg-config opencv --cflags --libs`
LIBS=$(OPENCV)
TARGET=main

SRC=$(shell ls *.cpp)

all: $(TARGET)

$(TARGET): $(SRC)
		$(CXX) $(CXXFLAGS) $(LIBS) $+ -o $@

clean:
		$(RM) -r *.o *.d *.dSYM *.txt $(TARGET)

.PHONY: all clean
