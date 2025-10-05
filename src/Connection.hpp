#ifndef CONNECTIONHEADERDEF
#define CONNECTIONHEADERDEF

#include <tuple>

struct Connection
{
    std::tuple<int, int> ElmConnect;
    std::tuple<int, int> EdgeConnect;
};

#endif