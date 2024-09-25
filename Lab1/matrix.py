"""
This module contains a Vector class with some general matrix operations as well as a Vector2 class and a Vector3 class
with specific methods and properties.

This is a working file that has been used to show different concepts, whatworks, and sometimes what does not.

You should not trust this very much.

DO NOT SHARE THIS FILE WITH ANYONE ELSE!
"""

import math
import vector

class Matrix(object):
    """This class contains matrix arithmatic and comparison operations."""
    def __init__(self, *args):
        """
        The constructor for the Matrix class.
        :param *args: a variable length list of Vectors
        :return: N/A for constructors
        """

        self.data = []
        if isinstance(args[0], vector.Vector):
            cols = args[0].dim
        else:
            raise TypeError('Arguments must be vectors')
        num_rows = 0
        for arg in args:
            if isinstance(arg, vector.Vector) and arg.dim == cols:
                self.data.append(arg.copy())
                num_rows += 1
            else:
                raise TypeError("Arguements must be vectors of the same dimension.")
        self.num_cols = args[0].dim
        self.num_rows = num_rows

    def copy(self):
        """
        Return a deep copy of an instance of the Vector class.
        :param: N/A
        :return: A deep copy of the Matrix
        """
        temp = []
        for arg in self.data:
            row = arg.copy()
            temp.append(row)
        return Matrix(*temp)

    def __str__(self):
        """
        Return a formatted Matrix
        Do not call this method directly.
        :return: a formatted string for use with print.
        """
        if len(self.data) == 1:
            values0 = ''
            for value in self.get_row(0).data:
                values0 = values0 + str(value) + ' '
            temp = '/ ' + str(values0) + '\ \n' 
        else:
            values0 = ''
            for value in self.get_row(0).data:
                values0 = values0 + str(value) + ' '
            temp = '/ ' + str(values0) + '\ \n' 
            for i in range(1, len(self.data) - 1):
                values1 = ''
                for value in self.get_row(i).data:
                    values1 = values1 + str(value) + ' '
                temp = temp + '| ' + str(values1) + '|\n'   
            values = ''
            for value in self.get_row(len(self.data)-1).data:
                values = values + str(value) + ' '
            temp = temp + '\ ' + str(values) + '/\n'

        return temp

    def __getitem__(self, location):
        """
        Return the value at the given Matrix row and column.
        :param location: row number and column number of the desired value
        :return: The float value at the position row,column
        """
        row = location[0]
        column = location[1]
        line = self.data[row].copy()
        value = line[column]
        return value

    def __setitem__(self, location, value):
        """
        Replaces value at indicated row and column with indicated value.
        :param location: indicated row and column of position to change
        :param value: new value to put into indicated position
        :return: None.  Changes value inside of list.append
        """
        row = location[0]
        column = location[1]
        self[row,column]
        if isinstance(value,(int,float)):
            self.data[row][column] = float(value)
        else:
            raise TypeError("A Matrix can only have numerical entries.")

    def get_row(self, row):
        """
        Returns indicated row of the Matrix as a Vector.
        :param row: The row of the Matrix to return
        :return: Vector form of indicated row.
        """
        return self.data[row]

    def get_column(self, column):
        """
        Returns indicated column of the Matrix as a Vector.
        :param column: The column of Matrix to return
        :return: Vector form of indicated column
        """
        vect = []
        for i in range(self.num_rows):
            row = self.get_row(i).copy()
            vect.append(row[column])
        return vector.Vector(*vect)

    def set_row(self, index, newrow):
        """
        Replace an existing row with a new Vector
        :param index: Row to replace
        :param vector: new vector 
        :return: None. Changes vector in list.
        """
        self.get_row(index).copy()
        if isinstance(newrow,(vector.Vector)):
            if newrow.dim == self.get_row(index).dim:
                self.data[index] = newrow
            else:
                raise TypeError('All rows must have same dimension.  New row must have dimension of {self[index].dim}.')
        else:
            raise TypeError('Matrix entries must be Vectors.')

    def set_column(self, index, newcol):
        """
        Replace an existing column with values from a new Vector.
        :param index: Indicates the column to change.
        :param vector: Gives the new values for the column
        :return: None. Changes values in each Vector in list.
        """
        if isinstance(newcol,(vector.Vector)):
            if newcol.dim == self.num_rows:
                for i in range(newcol.dim):
                    self[i,index] = newcol[i]    
            else:
                raise TypeError('Dimension mismatch, new column must have dimension {self.num_rows}.')
        else:
            raise TypeError('Matrix entries must be Vectors.')

    def __add__(self, other):
        """
        Overload the + operator. Return the sum of self and another Matrix if dimesions match.
        :param other: A Matrix of the same dimensions as self.
        :return: The Matrix sum of self and other.
        """
        if not isinstance(other, Matrix):
            raise TypeError('Can only add Matrices to other Matrices.')
        if self.num_rows != other.num_rows or self.get_row(0).dim != other.get_row(0).dim:
            raise TypeError('Matrices must have the same dimensions.')
        else:
            temp = []
            for i in range(self.num_rows):
                row = self.get_row(i).copy() + other.get_row(i).copy()
                temp.append(row) 
        return Matrix(*temp)

    def __sub__(self, other):
        """
        Overload the - operator.  Return the difference of self and Matrix other if the dimensions match.
        :param other: A Matrix of the same dimensions as self.
        :return: The Matrix difference of self and other.
        """
        if not isinstance(other, Matrix):
            raise TypeError('Can only subtract Matrices from other Matrices.')
        if self.num_rows != other.num_rows or self.data[0].dim != other.data[0].dim:
            raise TypeError('Matrices must have the same dimensions.')
        temp = []
        for i in range(self.num_rows):
            temp.append(self.get_row(i).copy() - other.get_row(i).copy()) 
        return Matrix(*temp)

    def __mul__(self, other):
        """
        Overload the * operator. Return the product of a scalar and a Matrix, the product of two Matrices, or the product of a Matrix and a Vector.
        :param other: can be a int, float, Matrix or Vector
        :return: The Matrix when multiplied by other.
        """
        if isinstance(other,(int,float)):
            temp = []
            for arg in self.data:
                temp.append(arg*other)
            return Matrix(*temp)
        elif isinstance(other,(Matrix)):
            if self.num_cols != other.num_rows:
                raise TypeError("To multiply matrices the number of rows in the first matrix must equal the number of columns in the second.")
            else:
                temp = []
                for i in range(self.num_rows):
                    row = []
                    for j in range(other.num_cols):
                        entry = 0
                        for k in range(self.num_rows):
                            entry += self[i,k] * other[k,j]
                        row.append(entry)                        
                    temp.append(vector.Vector(*row))
            return Matrix(*temp)   
        elif isinstance(other,(vector.Vector)): 
            if self.num_rows != other.dim:
                raise TypeError("Vector must be same dimension as # of rows in Matrix.")
            else:
                if self.num_rows != other.dim:
                    raise TypeError("To multiply matrices the number of rows in the first matrix must equal the number of columns in the second.")
                else:
                    temp = []
                    for i in range(other.dim):
                        temp.append(vector.Vector(other[i]))
                    othernew = Matrix(*temp)
                    
                return self * othernew

    def __rmul__(self, other):
        """
        Overload the * operator when the Matrix is on the right.
        :param other: int,float, or Vector value
        :return: The Matrix multiplied by the scalar or Vector
        """
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, (vector.Vector)):
            temp = []
            for i in range(0,1):
                row = []
                for j in range(self.num_cols):
                    entry = 0
                    for k in range (other.dim):
                        entry += (self[k,j]*other[k])
                    row.append(entry)
                temp.append(vector.Vector(*row))
        return Matrix(*temp)

    def __neg__(self):
        """
        Negate a Matrix
        :param: A Matrix instance
        :return: The negative of the Matrix
        """
        return (-1)*self

    def __eq__(self, other):
        """
        Overload the == operator.
        Return boolean inicating whether other is a Vector equal to self.
        :param other: a Matrix
        :return: If all parameters are equal, True, Otherwise, False.
        """
        if not isinstance(other, Matrix) or self.num_rows != other.num_rows or self.num_cols != other.num_cols:
            return False
        for i in range(self.num_rows):
            if self.get_row(i) != other.get_row(i):
                return False
        return True

    def det(self):
        """Return the determinant of a given 2x2, 3x3, or 4x4 Matrix."""
        if self.num_rows != self.num_cols or self.num_rows > 4 or self.num_rows < 2:
            raise TypeError("Matrix must be a 2x2, 3x3 or 4x4 Matrix.")
        else:
            if self.num_rows == 2:
                determine = (self[0,0]*self[1,1]) - (self[0,1]*self[1,0])
                return determine
            if self.num_rows == 3:
                determine = (self[0,0]*(self[1,1]*self[2,2]-self[2,1]*self[1,2]))-(self[0,1]*(self[1,0]*self[2,2]-self[1,2]*self[2,0]))+(self[0,2]*(self[1,0]*self[2,1]-self[1,1]*self[2,0]))
                return determine
            if self.num_rows == 4:
                A2 = self * self
                A3 = A2 * self
                A4 = A2 * A2
                determine = (1/24)*((trace(self))**4 - (6*(trace(self)**2)*trace(A2)) + 3*((trace(A2))**2) + 8*trace(self)*trace(A3) - 6*trace(A4))
                return determine

    def transpose(self):
        """Return the transpose of a given Matrix."""
        veclist = []
        for i in range(self.num_cols):
            temp =[]
            for j in range(self.num_rows):
                row = self.get_row(j)
                temp.append(row[i])
            veclist.append(vector.Vector(*temp))
        return Matrix(*veclist)

def identity(rows):
    """Return an Identity Matrix with the given number of rows."""
    veclist = []
    for i in range(rows):
        temp = []
        for j in range(rows):
            if i == j:
                temp.append(1)
            else:
                temp.append(0)
        veclist.append(vector.Vector(*temp))
    return Matrix(*veclist)

def zero(rows, columns):
    """Return a rows x columns Matrix with all zeros."""
    veclist = []
    for i in range (rows):
        temp = []
        for j in range(columns):
            temp.append(0)
        veclist.append(vector.Vector(*temp))
    return Matrix(*veclist)

def ones(rows, columns):
    """Return a rows x columns Matrix with all ones."""
    veclist = []
    for i in range (rows):
        temp = []
        for j in range(columns):
            temp.append(1)
        veclist.append(vector.Vector(*temp))
    return Matrix(*veclist)
            
def trace(self):
    """ Return the trace of a given square matrix, return error if not a square matrix."""
    if self.num_rows != self.num_cols:
        raise TypeError("Can only find Trace of a square Matrix.")
    else:
        temp = 0
        for i in range(self.num_rows):
            temp += self[i,i]
        return temp

def inverse(self):
    """Return the inverse of a 2x2, 3x3, or 4x4 Matrix."""
    if self.num_rows != self.num_cols or  self.num_rows < 1 or self.num_rows > 4:
        raise TypeError("Matrix must be a square matrix (2x2,3x3,4x4).")
    if self.det() == 0:
        raise ValueError(f"Inverse does not exist, determinate = 0.")
    A = self
    A2 = self * self
    A3 = A2 * self
    if self.num_rows == 2:
        B = Matrix(vector.Vector(self[1,1],(-1)*self[0,1]),vector.Vector((-1)*self[1,0],self[0,0]))
        inverse = (1/A.det())* B
        return inverse
    if self.num_rows == 3:
        inverse = (1/A.det())* (.5*((trace(A))**2 - trace(A2))*identity(3) - (self * trace(A)) + A2)
        return inverse
    if self.num_rows == 4:
        inverse = (1/A.det()) * ((1/6)*((trace(A))**3 - 3*(trace(A))*(trace(A2)) + 2*(trace(A3)))*(identity(4)) - .5*A*((trace(A))**2 - trace(A2)) + A2*trace(A) - A3)
        return inverse

def rotate(angle):
    """Creates the 2d Rotation Matrix for the given angle.
    :param angle: input angle in radian form.
    :return: 2x2 matrix to produce the given rotation.
    """
    rot_mat = Matrix(vector.Vector(math.cos(angle),(-1)*math.sin(angle)), vector.Vector(math.sin(angle),math.cos(angle)))
    return rot_mat

def hg(entry):
    """Return homogeneous coordinates for a given Vector or Matrix."""
    if isinstance(entry, Matrix):
        temp = []
        for i in range(entry.num_rows):
            row = entry.get_row(i)
            newrow = []
            for j in range(row.dim):
                newrow.append(row[j])
            newrow.append(1)
            temp.append(vector.Vector(*newrow))
        return Matrix(*temp)
    elif isinstance(entry, (vector.Vector)):
        temp = []
        for i in range(entry.dim):
            temp.append(entry[i])
        temp.append(1)
        return Matrix(*temp)
    else:
        raise TypeError("This function accepts only a Vector or a Matrix.")

def translate(*args):
    """
    Returns a Translation Matrix of the given dimension, with the given translation.
    :param change: First entry of list, is the dimention of the translation matrix, other items are the amount of translation in each direction.
    """
    dimension = args[0]
    temp = []
    for i in range(len(args)):
        row = []
        if i == dimension - 1:
            for j in range(dimension):
                if i != j:
                    row.append(args[j+1])
                else:
                    row.append(1)
        if i != dimension-1:
            for j in range(dimension):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
        temp.append(vector.Vector(*row))
    return Matrix(*temp)  

def project(dimension):
    """Returns a projection matrix to project from 4D to 3D or from 3D to 2D."""
    temp = []
    for i in range(dimension + 1):
        row = []
        for j in range(dimension):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        temp.append(vector.Vector(*row))
    return Matrix(*temp)
        
def strass(A, B, n_min=2):
    """
    Return product of A and B using Straussen Multiplication.
    :param A: left n x n matrix to multiply
    :param B: right n x n matrix to multiply
    :param n_min: integer value at which traditional matrix multiplication is used
    :return: matrix product
    """

    if isinstance(A, Matrix) and isinstance(B, Matrix) and A.num_rows == A.num_cols \
    and B.num_rows == B.num_cols and A.num_rows == B.num_rows:
        if math.log2(A.num_rows).is_integer():
            n = A.num_rows
        #else: We will fix the matrices that do not have n equal to a power of 2 in a bit.
    else:
        raise TypeError('The operands must be square matrices of the same dimension.')
    if n <= n_min:
        return A * B
    m = n/2



    




                    
