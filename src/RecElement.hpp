#ifndef RECELEMENTHEADERDEF
#define RECELEMENTHEADERDEF

#include <iostream>
#include <vector>
#include <array>
#include "Edge.hpp"

// SquareElement class represents a square element in a grid
// It contains a list of edges that define the boundaries of the square element
// The edges are defined by their endpoints in 2D space (x, y coordinates)
// Later add more methods including Jacobian, area, shape functions, etc.
class RecElement
{
    private:
        std::vector<Edge> Edge_list;

    public:
        // Custom Constructor
        RecElement(double ax, double bx, double ay, double by);
    friend class SquareGrid; // Allow SquareGrid to access private members

        // Method to get the list of edges
    const std::vector<Edge>& getEdges() const { return Edge_list;};
};
#endif