#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>
#include <iostream>
#include <Eigen/Dense>


int main(int argc, char* argv[])
{
    // Prepare an Eigen vector
    Eigen::VectorXd vec(5);
    vec << 0.0, 1.1, 2.2, 3.3, 4.4;

    // Create HDF5 file and dataset
    HighFive::File file("vec.h5", HighFive::File::Overwrite);
    auto dataset = file.createDataSet<double>(
        "/eigen_vec",
        HighFive::DataSpace::From(vec)
    );

    // Write and close
    dataset.write(vec);

    // Re-open and read back
    HighFive::File file_in("vec.h5", HighFive::File::ReadOnly);
    auto ds_in = file_in.getDataSet("/eigen_vec");
    Eigen::VectorXd vec2;
    ds_in.read(vec2);

    // Verify
    std::cout << "Read back: " << vec2.transpose() << "\n";
    return 0;
}