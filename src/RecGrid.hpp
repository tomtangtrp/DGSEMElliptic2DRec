#ifndef RECGRIDHEADERDEF
#define RECGRIDHEADERDEF

#include "Connection.hpp"
#include "ConnectionTable.hpp"

class RecGrid
{
    private:
        // SquareGrid of [axL, bxL] x [ayL, byL]
        double axL;
        double bxL;
        double ayL;
        double byL;
        // 
        int nCol; // Nel_x = nCol;
        int nRow; // Nel_y = nRow;
        // Connection table for the structured square grid
        ConnectionTable RecConnectionTable; 
    public:
        // Custom constructor for [0,1]x[0,1]
        RecGrid(int Nel_x, int Nel_y);
        // Custom constructor
        RecGrid(double ax, double bx, double ay, double by, int Nel_x, int Nel_y);
        ConnectionTable getRecConnectionTable();
    

};
#endif
