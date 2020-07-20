from libc.stdint cimport int64_t
from libcpp cimport nullptr
from libcpp.memory cimport shared_ptr

from Index cimport Index

cdef class IndexWrapper:
    #Usage of this class property is strictly prohibited
    cdef shared_ptr[Index] __impl

    #Constructor that allow zero cost copy
    def __cinit__(self, IndexWrapper other = None):
        if other is not None:
            self.__impl = shared_ptr[Index](other.__impl) 

    #Usage of this method is prohibited for outer usage
    cdef Index* _get_raw(self):
        cdef Index* raw_ptr = self.__impl.get() 
        if raw_ptr is nullptr:
            raise MemoryError
        return raw_ptr

    #Property that stores dimensionality of vectors 
    @property.getter
    def d(self):
        return self._get_raw().d

    #Property that stores number of vectors in train dataset 
    @property.getter
    def ntotal(self):
        return self._get_raw().ntotal

    #Adds vectors to storage
    cdef add(self, int64_t n, const float* x):
        return self._get_raw().add(n, x)

    #Adds vectors to storage and performs training
    cdef train(self, int64_t n, const float* x):
        return self._get_raw().train(n, x)

    #def train(self, np.ndarray[float, ndim = 1, mode = "c"])

    #Adds vectors with predefined indices
    cdef add_with_ids(self, int64_t n, const float* x, const int64_t* xids):
        return self._get_raw().add_with_ids(n, x, xids)
    
    #Deletes all vectors from Index
    cdef reset(self):
        return self._get_raw().reset()

    #It allow us to create instance with direct forwarding of ptr
    #Also it is prohibited for outer usage
    @staticmethod
    cdef IndexWrapper _from_raw_ptr(Index* raw_ptr):
        cdef IndexWrapper result = IndexWrapper()
        result.__impl = shared_ptr[Index](raw_ptr)
        return result