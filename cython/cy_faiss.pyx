from libc.stdint cimport int64_t
from libcpp cimport nullptr
from libcpp.memory cimport shared_ptr

cdef extern from "faiss/Index.h" namespace "faiss":
    cdef cppclass Index:
        int64_t d
        int64_t ntotal
        void add(int64_t n, const float* x) 
        void train(int64_t n, const float* x, int64_t k, float* distances, int64_t labels)


cdef class IndexWrapper:
    cdef shared_ptr[Index] impl

    def __cinit__(self, IndexWrapper other = None):
        if other is not None:
            self.impl = shared_ptr[Index](other.impl) 

    cdef Index* get_raw(self):
        cdef Index* raw_ptr = self.impl.get() 
        if raw_ptr is nullptr:
            raise MemoryError
        return raw_ptr

    @property
    def d(self):
        return self.get_raw().d

    @staticmethod
    cdef IndexWrapper from_raw_ptr(Index* raw_ptr):
        cdef IndexWrapper result = IndexWrapper()
        result.impl = shared_ptr[Index](raw_ptr)
        return result