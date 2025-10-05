#include <iostream>
#include "SquareGrid.hpp"
#include "ConnectionTable.hpp"

int main(int argc, char* argv[])
{
    int Nel_x = 2;
    int Nel_y = 2;
    SquareGrid mSquareGrid(Nel_x, Nel_y);
    ConnectionTable mSqaureConnectionTable = mSquareGrid.getSquareConnectionTable();
    mSqaureConnectionTable.printConnectionTable();

    return 0;
}