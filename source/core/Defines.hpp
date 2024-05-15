#pragma once

#define fn auto

// Allocates an uninitialized array on the stack.
#define STACK_ALLOCATE_UNINIT(Type, count) (static_cast<Type*>(alloca(count * sizeof(Type))))

#define STACK_ALLOCATE_ZEROED(Type, count) (static_cast<Type*>(memset(alloca(count * sizeof(Type)), 0, count * sizeof(Type))))

#ifdef NDEBUG
#define DEBUG_BUILD 0
#else
#define DEBUG_BUILD 1
#endif

// @TODO: Other build defines.