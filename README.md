# DGSEMElliptic2DRec
practice project on DGSEM 2D rectangular grid for Elliptic PDEs, using Eigen's intel Pardiso sparse direct solver with OPENMP

# Dependencies
### install highfive
```bash
cd /usr/local/include
git clone --recursive https://github.com/BlueBrain/HighFive.git
cd HighFive
cmake -DCMAKE_INSTALL_PREFIX=build/install -DHIGHFIVE_USE_BOOST=Off -B build .
cmake --build build --parallel
cmake --install build
```

### ubuntu
```bash
sudo apt install libeigen3-dev
sudo apt install libhdf5-dev
```

### gadi
```bash
module load eigen/3.3.7
module load hdf5/1.10.7
module load intel-mkl/2025.2.0
module load intel-compiler-llvm/2025.2.0
```





# Build
```bash
make testSolver_sp
```

# Run
```bash
./testSolver_sp n SIPG 3 10 256 256 0 6.283185307179586 0 6.283185307179586 "L=D,R=D,B=N,T=N" cases/testcase1.json
```
