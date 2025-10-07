# make -j$(nproc) testSolver_sp for parallel build
# (0) Build dirs
SRC_DIR = src
OBJ_DIR = build

# (1) Compiler and flags
include machine.mk

# (2) Sources, objects, and executables
SRC_CODES     = gllQuadrature.cpp RecElement.cpp RecGrid.cpp JsonCase.cpp testGLL.cpp testEigen.cpp testMeshGrid.cpp testConnectionTable.cpp testSquareGrid.cpp testHighFive.cpp testSolver.cpp testSolver_sp.cpp testSolver_sp_eigen.cpp testSolver_spCG.cpp
SRCS = $(addprefix $(SRC_DIR)/, $(SRC_CODES))
# use substitution referece to convert each filename.cpp to filename.o
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGETS  = testGLL testEigen testMeshGrid testConnectionTable testSquareGrid testHighFive testSolver testSolver_sp testSolver_sp_eigen testSolver_spCG

# (3) Declares all and clean as phony targets, meaning they aren’t real files for makefile.
# Ensures Make will always run their recipes when you invoke them.
.PHONY: all clean

#!!! always put default  target first so make will build if there's no input param after make
all: $(TARGETS)

# (4) Link testGLL
# $@ expands to the target names, $@ = testGLL
# $^ is all prerequisites testGLL.o gllQuadrature.o
testGLL: testGLL.o gllQuadrature.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Link testEigen
testEigen: testEigen.o gllQuadrature.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Link testMeshGrid
testMeshGrid: testMeshGrid.o gllQuadrature.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

testConnectionTable: testConnectionTable.o gllQuadrature.o RecGrid.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

testSquareGrid: testSquareGrid.o RecGrid.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

testHighFive: testHighFive.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

testSolver: testSolver.o gllQuadrature.o RecGrid.o RecElement.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

testSolver_sp: $(SRC_DIR)/testSolver_sp.o $(SRC_DIR)/gllQuadrature.o $(SRC_DIR)/RecGrid.o $(SRC_DIR)/RecElement.o $(SRC_DIR)/JsonCase.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

testSolver_sp_eigen: testSolver_sp_eigen.o gllQuadrature.o RecGrid.o RecElement.o JsonCase.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

testSolver_spCG: testSolver_spCG.o gllQuadrature.o RecGrid.o RecElement.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# (5) Compile .cpp → .o
# pattern rule %.0 for object file, %.cpp for source file
# $< is the first prerequisite afeter: (%.cpp)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# (6) Automatically delete object files after building, this will destroy incremental build, only keep during testing where codes are expected to change a lot
.INTERMEDIATE: $(OBJS)

# Manual clean (if you ever want to remove binaries too)
clean:
	rm -f $(TARGETS)
