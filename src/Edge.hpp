#ifndef EDGEHEADERDEF
#define EDGEHEADERDEF

#include <array>

class Edge
{
    private:
        std::array<double, 2> x_endPoints; // x-coordinates of the edge endpoints
        std::array<double, 2> y_endPoints; // y-coordinates of the  edge endpoints
        bool is_bdry = false; // flag to indicate if the edge is a boundary edge
        int edge_lid = -1; // local unique identifier for the edge

    public:
        // Custom Constructor
        Edge(double ax, double bx, double ay, double by, int local_id) { 
            edge_lid = local_id;
            x_endPoints[0] = ax; x_endPoints[1] = bx;
            y_endPoints[0] = ay; y_endPoints[1] = by;
        };
        // Later build this into SquareGrid
        void check_bdry(double axL, double bxL, double ayL, double byL) {
            double tol = 1e-14; // tolerance for boundary check
            if (std::abs(y_endPoints[0]-ayL)<tol && std::abs(y_endPoints[1]-ayL)<tol) {// This is a bottom edge
                is_bdry = true;
            }
            else if (std::abs(x_endPoints[0]-axL)<tol && std::abs(x_endPoints[1]-axL)<tol) { // This is a left edge
                is_bdry = true;
            }
            else if (std::abs(y_endPoints[0]-byL)<tol && std::abs(y_endPoints[1]-byL)<tol) { // This is a top edge
                is_bdry = true;
            }
            else if (std::abs(x_endPoints[0]-bxL)<tol && std::abs(x_endPoints[1]-bxL)<tol) { // This is a right edge
                is_bdry = true;
            }
            else {
                is_bdry = false;
            }
            
        };

        // Getters for is_bdry and edge_lid
        bool get_is_bdry() const { return is_bdry; };
        int get_edge_lid() const { return edge_lid; };
};      
#endif