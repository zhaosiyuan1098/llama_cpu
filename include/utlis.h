#ifndef UTILS_H
#define UTILS_H


template <typename T>
void read_to_array(const char* path, T* array, int size);

template <typename T>
void allocate_aligned_memory(T*& ptr, size_t size);

#endif
