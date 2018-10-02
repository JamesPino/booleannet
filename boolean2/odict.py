#
# a dictionary-like class that maintains the order of insertion
# 
# based on a recipe by Igor Ghisi located at
#
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/496761
#
# from collectiionsUserDict import DictMixin
from collections import UserDict


class odict(UserDict):
    """
    >>> o = odict()
    >>> o[2]=20 ; o[1]=10
    >>> o.keys()
    [2, 1]
    >>> o.values()
    [20, 10]
    >>> o.items()
    [(2, 20), (1, 10)]
    >>> [ x for x in o ]
    [2, 1]
    >>>
    >>> d = dict()
    >>> d[2]=20 ; d[1]=10
    >>> d.keys()
    [1, 2]
    >>> d.values()
    [10, 20]
    >>> d.items()
    [(1, 10), (2, 20)]
    >>> [ x for x in d ]
    [1, 2]

    """
    def __init__(self, **kwds):
        self._keys = []
        self.data = {}
        for key, value in kwds.items():
            self[key] = value
        
    def __setitem__(self, key, value):
        if key not in self.data:
            self._keys.append(key)
        self.data[key] = value
        
    def __getitem__(self, key):
        return self.data[key]
    
    def __delitem__(self, key):
        del self.data[key]
        self._keys.remove(key)
        
    def keys(self):
        return list(self._keys)
    
    def copy(self):
        copyDict = odict()
        copyDict.data = self.data.copy()
        copyDict._keys = self._keys[:]
        return copyDict

if __name__ == '__main__':
    import doctest
    doctest.testmod()