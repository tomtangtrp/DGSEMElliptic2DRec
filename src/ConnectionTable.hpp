#ifndef CONNECTIONTABLEHEADERDEF
#define CONNECTIONTABLEHEADERDEF

#include "Connection.hpp"
#include "iostream"
#include <vector>

class ConnectionTable
{   
    private:
        std::vector<Connection> ConnectionTable;
    public:
        void add_connection(Connection connect){ConnectionTable.push_back(connect);};
        void printConnectionTable(){
            std::cout << "ConnectionTable:" << "\n";
            for (int i=0; i<ConnectionTable.size(); i++)
            {   
                
                std::cout << "[ Element Connected: (" << std::get<0>(ConnectionTable[i].ElmConnect) << ", " << std::get<1>(ConnectionTable[i].ElmConnect) << "), "
                << " Edge Connected: (" << std::get<0>(ConnectionTable[i].EdgeConnect) << ", " << std::get<1>(ConnectionTable[i].EdgeConnect) << ")]" << "\n";

            }
            std::cout<< std::endl;
        };
        // Overloading the [] operator to access ConnectionTable elements
        Connection& operator[](int index) { return ConnectionTable[index]; };

        int getSize() const { return ConnectionTable.size(); };
};
#endif