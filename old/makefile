# Actually faster for small problems, but not for large ones
# this is because For small matrices/vectors, 
# Eigen’s inlined, vectorized kernels (with -Ofast) beat the overhead of calling into BLAS.
gllQuadrature.o : gllQuadrature.hpp gllQuadrature.cpp ./include/Legendre.hpp
	g++ -I./include -O -c gllQuadrature.cpp

SquareElement.o : SquareElement.hpp SquareElement.cpp Edge.hpp
	g++ -O -c SquareElement.cpp

SquareGrid.o : SquareGrid.hpp SquareGrid.cpp SquareElement.hpp ConnectionTable.hpp Connection.hpp
	g++ -O -c SquareGrid.cpp

testSolver.o : testSolver.cpp gllQuadrature.hpp ./include/Legendre.hpp ./include/MeshGrid2D.hpp SquareElement.hpp SquareGrid.hpp ConnectionTable.hpp NumUtilities.hpp
	g++ -I/usr/include/eigen3 -I/usr/include/hdf5/serial -I/usr/local/include/HighFive/include -I./include -O -c testSolver.cpp

testGLL.o : testGLL.cpp gllQuadrature.hpp Legendre.hpp
	g++ -O -c testGLL.cpp

testEigen.o : testEigen.cpp gllQuadrature.hpp
	g++ -I/usr/include/eigen3 -I./include -O -c testEigen.cpp


# !!! include hpp in compiling to ensure header changes impact on cpp files
# !!! but without hpp will also work (make scan the current dir and look for hpp by #include "Legendre.hpp" in codes) 

# without gllQuadrature.hpp also works
#testEigen.o : testEigen.cpp


testGLL : testGLL.o gllQuadrature.o 
	g++ -O -o testGLL testGLL.o gllQuadrature.o 

testEigen : testEigen.o gllQuadrature.o
	g++ -O -o testEigen testEigen.o gllQuadrature.o

testSolver : testSolver.o gllQuadrature.o SquareElement.o SquareGrid.o 
	g++ -O -o testSolver testSolver.o gllQuadrature.o SquareElement.o SquareGrid.o -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5

clean:
	rm -r -f *.o