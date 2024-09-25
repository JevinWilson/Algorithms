"""
This module contains a Vector class with some general vector operations as well as a Vector2 class and a Vector3 class
with specific methods and properties. It also contains functions for dot product, cross product, and conversion
from polar coordinates to a two-dimensional Vector2.
"""

# This Vector class is mostly correct. Part of it was used to show various things that could go wrong. I think I
# fixed all the errors but DO NOT trust it completely until you have checked it.

# DO NOT share this file with anyone else.

import  math

class Vector(object):
    def __init__(self, *args):
        """
        The constructor for the Vector class.
        :param *args: A variable length argument list of ints or floats
        :return: N/A for constructors
        """
        self.data = []
        for value in args:
            if isinstance(value,(float, int)):
                self.data.append(float(value))
            else:
                raise TypeError('Only integer or float values can be accepted.')

        self.dim = len(self.data)

        if self.dim == 2:
            self.__class__ = Vector2
        elif self.dim == 3:
            self.__class__ = Vector3

    def __getitem__(self, index):
        """
        Return the value at the given index.
        :param index: An integer index
        :return: The float value at position index
        """
        if isinstance(index, int):
            if -self.dim <= index < self.dim:
                return self.data[index]
            return IndexError
        return TypeError('The position must be an integer index.')

    def __setitem__(self, index, value):
        """
        Update the value at the given index with the specified value.
        :param index: integer index to be updated
        :param value: new float value value 
        :return: Returns None. Changes the list index with the value.
        """
        if isinstance(value, (float, int)):
            self.data[index] = float(value)
        else:
            raise TypeError('Only integer or float values can be accepted.')
        
    def __str__(self):
        """
        Return a formatted string of the form <Vector{dim}: {data}> for use with print().
        Do not call this method directly.
        :return: a formatted string for use with print
        """
        data_string = f'<Vector{self.dim}:'
        for i in range(self.dim):
            if i < self.dim - 1:
                data_string += ' ' + str(self[i]) + ','
            elif i == self.dim - 1:
                data_string += ' ' + str(self[i]) + '>'
        return data_string

    def __len__(self):
        """
        Return the number of elements in an instance of the Vector class.
        :param: N/A
        :return: Returns length of self.data
        """
        return len(self.data)

    def __eq__(self, other):
        """
        Overload == operator.
        Return boolean indicating whether other is a Vector equal to self.
        :param other: A Vector
        :return: If all the the vector parameters are equal, returns True. Otherwise, False
        """
        if isinstance(other, Vector) and self.dim == other.dim:
            for i in range(self.dim):
                if self[i] != other[i]:
                    return False
            return True
        return False

    def copy(self):
        """
        Return a deep copy of an instance of the Vector class.
        :param: N/A
        :return: A deep copy of the Vector
        """
        # Note: This could be completed with the single line return Vector(*self.data)
        # but the approach used here carries over to other deep copy scenarios like a Matrix class. 
        temp = []
        for value in self.data:
            temp.append(value)
        return Vector(*temp)

    def __mul__(self, other):
        """
        Overload the * operator.
        Return the product of a Vector and a scalar, or NotImplemented for other data types.
        :param other: int or float. Other data types NotImplemented
        :return: The vector multiplied on the right by a scalar
        """
        if isinstance(other, (float,int)):
            v = self.copy()
            for i in range(self.dim):
                v[i] *= other
            return v
        else:
            #raise TypeError("Can only multiply a vector by a float or integer scalar.")
            return NotImplemented        

    def __rmul__(self, other):
        """
        Overload the * operator when the Vector is on the right.
        :param other: int or float 
        :return: The vector multiplied on the left by a scalar 
        """
        return self * other

    def __add__(self, other):
        """
        Overload the + operator.
        Return the sum of self and a Vector other if the dimensions match.
        :param other: A Vector of the same dimension as self
        :return: The Vector sum of self and other
        """
        if isinstance(other, Vector) and self.dim == other.dim:
            v = self.copy()
            for i in range(self.dim):
                v[i] += other[i]
            return v
        else:
            raise TypeError('You can only add another Vector' + str(self.dim) + ' to this Vector' + str(self.dim)+ ' (You passed "' + str(other)+ '".)')
        
    def __sub__(self, other):
        """
        Overload the - operator.
        :param other: A Vector 
        :return: The Vector self - other
        """
        return self + -other
    
    def __neg__(self):
        """
        Negate a Vector.
        :param: A Vector instance
        :return: The negative of the Vector
        """
        return self * -1

    def __truediv__(self, other):
        """
        Overload the / operator.
        :param other: A float or an int
        :return: A Vector divided by the scalar other
        """
        if isinstance(other, (float,int)) and other != 0:
            v = self.copy()
            for i in range(self.dim):
                v[i] /= float(other)
            return v
        elif other == 0:
            raise ZeroDivisionError('Cannot divide a Vector by 0.')
        else:
            raise TypeError('Can only divide a Vector by a non-zero float or integer scalar.')

    def norm(self, p):
        """
        Return the p-norm of a Vector instance.
        :param: A positive number p
        :return: the corresponding p-norm
        """
        if isinstance(p, str) and p == 'infinity':
            value_list=[]
            for i in range(self.dim):
                value_list.append(abs(self[i]))
            return max(value_list)
        value = 0
        if isinstance(p, (int, float)) and p > 0:
            for i in range(self.dim):
                print(value)
                value += (abs(self[i]))**p
            return value**(1/p)
        else:
            raise ValueError('The p-norm cannot use negative numbers or zero.')  

    @property
    def mag(self):
        """
        Return the 2-norm of a Vector instance as a property. 
        :param: N/A 
        :return: The 2-norm of this vector        """
        return self.norm(2)

    @property
    def mag_squared(self):
        """
        Return the magnitude squared of a Vector instance as a property without using square roots. 
        :param: N/A
        :return: The square of the 2-norm without using any square roots
        """
        value = 0
        for i in range(self.dim):
            value+=(abs(self[i]))**2
        return value

    @property
    def normalize(self):
        """
        Return a unit vector parallel to a Vector instance.
        :param: N/A
        :return: A unit vector in the same direction as the current vector
        """
        return self/self.mag

    @property
    def is_zero(self):
        """
        Return boolean indicating whether the given Vector is identically the zero Vector.
        :param: N/A
        :return: True if this Vector is identically the zero Vector of the appropriate
                 dimension, False otherwise
        """
        for i in range(self.dim):
            if self[i] != 0:
                return False
        return True

    @property
    def i(self):
        """
        Return a tuple of the coordinates of the Vector converted to integers.
        :param: N/A
        :return: A tuple of the coordinates of this Vector converted to integers.
        """
        k = []
        for i in range(self.dim):
            k.append(int(self[i]))
        return tuple(k)
    
class Vector2(Vector):
    def __init__(self, x, y):
        """
        The constructor for the Vector2 class.
        :parameters: Floats or ints x and y
        :return: Creates a vector instance that is in Vector and Vector2 class
        """
        super().__init__(x, y)

    @property
    def x(self):
        """
        Return the x value of a Vector2.
        :param: N/A
        :return: The x component
        """
        return self[0]

    @x.setter
    def x(self, newvalue):
        """
        Change the x value of a Vector2.
        :param: The integer or float value to which x should be changed
        :return: Returns None. Changes x value to new value.
        """
        if isinstance(newvalue, (int, float)):
            self[0] = float(newvalue)
        else:
            raise TypeError('Only integer or float values can be accepted.')
        
    @property
    def y(self):
        """
        Return the y value of a Vector2.
        :param: N/A
        :return: The y component
        """
        return self[1]

    @y.setter
    def y(self, newvalue):
        """
        Change the y value of a Vector2.
        :param: The integer or float value to which y should be changed
        :return: Returns None. Changes y value to new value.
        """
        if isinstance(newvalue, (int, float)):
            self[1] = float(newvalue)
        else:
            raise TypeError('Only integer or float values can be accepted.')

    @property
    def degrees(self):
        """
        Return the degree measure of a traditional cartesian Vector2 in polar space.
        :parameter: N/A
        :return: The degree measure of this cartesian vector in polar space
        """
        return math.degrees(math.atan2(self.y, self.x))

    @property
    def degrees_inv(self):
        """
        Return the degree measure of a cartesian Vector2 in polar space with the y-value changed to account for pygame.
        :parameter: N/A
        :return: The degree measure of this cartesian vector in polar space flipped for pygame
        """
        return math.degrees(math.atan2(-self.y, self.x))

    @property
    def radians(self):
        """
        Return the radian measure of a traditional cartesian Vector2 in polar space.
        :parameter: N/A
        :return: The radian measure of this cartesian vector in polar space
        """
        return math.atan2(self.y, self.x)

    @property
    def radians_inv(self):
        """
        Return the radian measure of a cartesian Vector2 in polar space with the y-value changed to account for pygame.
        :parameter: N/A
        :return: The radian measure of this cartesian vector in polar space flipped for pygame
        """
        return math.atan2(-self.y, self.x)

    @property
    def perpendicular(self):
        """
        Return a Vector2 perpendicular to the given Vector2.
        :parameter: N/A
        :return: A Vector2 perpendicular to this Vector
        """
        return Vector(-self.y, self.x)
    
class Vector3(Vector):
    def __init__(self, x, y, z):
        """
        The constructor for the Vector3 class.
        :parameters: Floats or ints x, y, and z
        :return: Creates a vector instance that is in Vector and Vector3 class
        """
        super().__init__(x, y, z)

    @property
    def x(self):
        """
        Return the x value of a Vector3.
        :param: N/A
        :return: The x component
        """
        return self[0]

    @x.setter
    def x(self, new_value):
        """
        Change the x value of a Vector3.
        :param: The integer or float value to which x should be changed
        :return: Returns None. Changes x value to new value.
        """
        if isinstance(new_value, (int, float)):
            self[0] = float(new_value)
        else:
            raise TypeError('Only integer or float values can be accepted.')
        
    @property
    def y(self):
        """
        Return the y value of a Vector3.
        :param: N/A
        :return: The y component
        """
        return self[1]

    @y.setter
    def y(self, new_value):
        """
         Change the y value of a Vector3.
        :param: The integer or float value to which y should be changed
        :return: Returns None. Changes y value to new value.
        """
        if isinstance(new_value, (int, float)):
            self[1] = float(new_value)
        else:
            raise TypeError('Only integer or float values can be accepted.')

    @property
    def z(self):
        """
        Return the z value of a Vector3.
        :param: N/A
        :return: The z component
        """
        return self[2]

    @z.setter
    def z(self, new_value):
        """
        Change the z value of a Vector3.
        :param: The integer or float value to which z should be changed
        :return: Returns None. Changes z value to new value.
        """
        if isinstance(new_value, (int, float)):
            self[2] = float(new_value)
        else:
            raise TypeError('Only integer or float values can be accepted.')

#Functions
def dot(v1, v2):
    """
    Return the dot product of two vectors of the same dimension.
    :parameters: Two Vectors of the same dimension
    :return: The dot product of the two Vectors
    """  
    if isinstance(v1, Vector) and isinstance(v2, Vector) and v1.dim == v2.dim:
        scalar = 0
        for i in range(v1.dim):
            scalar += v1[i] * v2[i]
        return scalar
    else:
        raise TypeError('The dot product requires two Vectors of the same dimension.')
    
def cross(v1, v2):
    """
    Return the cross product of two three-dimensional vectors.
    :parameters: Two 3-dimensional vectors
    :return: A Vector3 giving the cross product of 3D vectors v and w
    """
    if isinstance(v1, Vector3) and isinstance(v2, Vector3):
        value1 = v1.y * v2.z - v1.z * v2.y
        value2 = v1.x * v2.z - v1.z * v2.x
        value3 = v1.x * v2.y - v1.y * v2.x
        return Vector(value1, -value2, value3)
    else:
        raise TypeError('Cross product is only valid for two 3D vectors.')

def polar_to_Vector2(r, angle, neg = False):
    """
    Return a Vector2 given a float or int radius and angle.
    :parameters: Float or int radius, float or int angle in radians, and a boolean option to negate y for pygame
    :return: A two dimensional vector at the x and y position
    """
    if isinstance(r, (int, float)) and isinstance(angle, (int, float)):
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        if neg == True:
            y = -y
        return Vector(x, y)
    else:
        raise TypeError('The input must be a radius and an angle.')


