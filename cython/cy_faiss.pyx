from libc.stdint inmport int64_t
from libcpp.memory inmport shared_ptr, nullptr

cdef extern from "faiss/Index.h" namespace "faiss":
    cdef cppclass Index:
        int64_t d
        int64_t ntotal
        void add(int64_t n, const float* x) 
        void train(int64_t n, const float* x, int64_t k, float* distances, int64_t labels)


cdef class IndexWrapper:
    cdef shared_ptr[Index] impl

    def __cinit__(self, Index* ptr):
        self.impl = shared_ptr[Index](ptr) 

    @property
    cdef Index* get_raw(self):
        if impl.get() is nullptr:
            raise MemoryError
        return impl.get()

    @property
    cdef int64_t d(self):
        return self.get_raw.d