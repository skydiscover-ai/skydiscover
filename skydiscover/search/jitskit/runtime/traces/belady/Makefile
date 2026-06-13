CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

belady_sim: belady_sim.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f belady_sim

.PHONY: clean
