#include <iostream>
#include "RecGrid.hpp"
#include "Connection.hpp"
#include "ConnectionTable.hpp"

// Custom Constructor on [0,1]x[0,1]
RecGrid::RecGrid(int Nel_x, int Nel_y)
{
    axL = 0.0;
    bxL = 1.0;
    ayL = 0.0;
    byL = 1.0;

    nCol=Nel_x;
    nRow=Nel_y;
}



//Custom Constructor on [ax, bx]x[ay, by]
RecGrid::RecGrid(double ax, double bx, double ay, double by, int Nel_x, int Nel_y)
{
    axL = ax;
    bxL = bx;
    ayL = ay;
    byL = by;

    nCol=Nel_x;
    nRow=Nel_y;
}

ConnectionTable RecGrid::getRecConnectionTable()
{   
    ConnectionTable SquareConnectionTable;
    for (int i=0; i<(nRow-1); i++)
    {
        for (int j=0; j<nCol; j++)
        {   
            // topEdge-bottomEdge connect
            int N_E = j + i*nCol;
            int N_E_topEdge = N_E + nCol;
            Connection top_bottom_connect;
            top_bottom_connect.ElmConnect=std::make_tuple(N_E, N_E_topEdge);
            top_bottom_connect.EdgeConnect=std::make_tuple(2,0);
            SquareConnectionTable.add_connection(top_bottom_connect);

            // rightEdge-leftEdge connect
            if (N_E < (i+1)*nCol-1)
            {
                int N_E_leftEdge = N_E+1;
                Connection right_left_connect;
                right_left_connect.ElmConnect=std::make_tuple(N_E, N_E_leftEdge);
                right_left_connect.EdgeConnect=std::make_tuple(1,3);
                SquareConnectionTable.add_connection(right_left_connect);
            }
        }
    }
    // Last row on sqauregrid:
    for (int i=0; i<nCol-1; i++)
    {
        int N_E_last = i+(nRow-1)*nCol;
        int N_E_last_leftEdge = (nRow-1)*nCol + (i+1);
        Connection right_left_connect;
        right_left_connect.ElmConnect=std::make_tuple(N_E_last, N_E_last_leftEdge);
        right_left_connect.EdgeConnect=std::make_tuple(1,3);
        SquareConnectionTable.add_connection(right_left_connect);
    }
    return SquareConnectionTable;
}

