from libc.stdint cimport int64_t
from libcpp cimport nullptr
from libcpp.memory cimport shared_ptr

cdef extern from "faiss/Index.h" namespace "faiss":
    cdef cppclass Index:
        int64_t d
        int64_t ntotal
        void add(int64_t n, const float* x)
        void add_with_ids(int64_t n, const float* x, const int64_t* xids)
        void train(int64_t n, const float* x) 
        void search(int64_t n, const float* x, int64_t k, float* distances, int64_t labels)
        void reset()