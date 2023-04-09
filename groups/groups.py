from numbers import Integral
import numpy as np


class Element:
    def __init__(self, group, value):
        group._validate(value)
        self.group = group
        self.value = value

    def __mul__(self, other):
        return Element(
            self.group,
            self.group.operation(self.value, other.value)
        )
    
    def __str__(self):
        return f"{self.value}_{self.group}"
    
    def __repr__(self):
        return f"{type(self).__name__}({self.group}, {self.value})"
    

class Group:
    """A base class containing methods common to many groups.
    
    Each subclass represents a family of parametrised groups.
    
    Parameers
    ---------
    n: int
        The primary group parameter, such as order or degree. The
        precise meaning of n changes from subclass to subclass.
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, value):
        """Create an element of the group."""
        return Element(self, value)
    
    def __repr__(self):
        """Return a string representation of the group."""
        return f"{type(self).__name__}({self.n})"
    
    def __str__(self):
        """Return a string representation of the group."""
        return f"{self.symbol}{self.n}"
    
class CyclicGroup(Group):
    """A cyclic group represented by integer addition modulo n."""
    symbol = "C"

    def _validate(self, value):
        """Check that value is an integer in the range [0, n)."""
        if not (isinstance(value, Integral) and
                0 <= value < self.n):
            raise ValueError("Element value must be an integer "
                             f"in the range [0, {self.n}).")
        
    def operation(self, a, b):
        """Add two integers modulo n."""
        return (a + b) % self.n

    
class GeneralLinearGroup(Group):
    """A general linear group represented by matrix multiplication."""
    symbol = "G"

    def _validate(self, value):
        """Check that value is a square matrix of size n x n."""
        if not (isinstance(value, np.ndarray) and
            value.shape == (self.n, self.n)):
            raise ValueError("Element value must be an array "
                             f"with shape ({self.n}, {self.n}).")
        
    def operation(self, a, b):
        """Multiply two matrices."""
        return a @ b