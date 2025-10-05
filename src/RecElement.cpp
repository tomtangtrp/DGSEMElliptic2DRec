#include <iostream>
#include <vector>
#include "RecGrid.hpp"
#include "RecElement.hpp"
#include "Edge.hpp"

RecElement::RecElement(double ax, double bx, double ay, double by) {
    // Initialize edges of the square element
    Edge edge0(ax, bx, ay, ay, 0); // Bottom edge
    Edge edge1(bx, bx, ay, by, 1); // Right edge
    Edge edge2(ax, bx, by, by, 2); // Top edge
    Edge edge3(ax, ax, ay, by, 3); // Left edge

    Edge_list.push_back(edge0);
    Edge_list.push_back(edge1);
    Edge_list.push_back(edge2);
    Edge_list.push_back(edge3);

}
