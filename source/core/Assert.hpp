#pragma once

#include "Defines.hpp"

#if DEBUG_BUILD
#define ASSERT(...) \
    { \
        if (!(__VA_ARGS__)) { \
            std::cerr << #__VA_ARGS__ << " evaluates to false!" << std::endl; \
            assert(false); \
        } \
    }
#else
#define ASSERT(...)
#endif
