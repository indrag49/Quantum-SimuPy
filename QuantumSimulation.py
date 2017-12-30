import numpy as np
import math
c=np.concatenate
o=np.outer

## Initialise the 1-Qubit system: {|0>, |1>}
Q0 = np.array([1., 0.])
Q1 = np.array([0., 1.])

## Initialise the 2-Qubit system: {|00>, |01>, |10>, |11>}
Q00 = c(o(Q0, Q0))
Q01 = c(o(Q0, Q1))
Q10 = c(o(Q1, Q0))
Q11 = c(o(Q1, Q1))

## 3-Qubit system: {|000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>}
Q000 = c(o(Q00, Q0))
Q001 = c(o(Q00, Q1))
Q010 = c(o(Q01, Q0))
Q011 = c(o(Q01, Q1))
Q100 = c(o(Q10, Q0))
Q101 = c(o(Q10, Q1))
Q110 = c(o(Q11, Q0))
Q111 = c(o(Q11, Q1))

## 4-Qubit system
Q0000 = c(o(Q000, Q0))
Q0001 = c(o(Q000, Q1))
Q0010 = c(o(Q001, Q0))
Q0011 = c(o(Q001, Q1))
Q0100 = c(o(Q010, Q0))
Q0101 = c(o(Q010, Q1))
Q0110 = c(o(Q011, Q0))
Q0111 = c(o(Q011, Q1))
Q1000 = c(o(Q100, Q0))
Q1001 = c(o(Q100, Q1))
Q1010 = c(o(Q101, Q0))
Q1011 = c(o(Q101, Q1))
Q1100 = c(o(Q110, Q0))
Q1101 = c(o(Q110, Q1))
Q1110 = c(o(Q111, Q0))
Q1111 = c(o(Q111, Q1))

## 5-Qubit system
Q00000 = c(o(Q0000, Q0))
Q00001 = c(o(Q0000, Q1))
Q00010 = c(o(Q0001, Q0))
Q00011 = c(o(Q0001, Q1))
Q00100 = c(o(Q0010, Q0))
Q00101 = c(o(Q0010, Q1))
Q00110 = c(o(Q0011, Q0))
Q00111 = c(o(Q0011, Q1))
Q01000 = c(o(Q0100, Q0))
Q01001 = c(o(Q0100, Q1))
Q01010 = c(o(Q0101, Q0))
Q01011 = c(o(Q0101, Q1))
Q01100 = c(o(Q0110, Q0))
Q01101 = c(o(Q0110, Q1))
Q01110 = c(o(Q0111, Q0))
Q01111 = c(o(Q0111, Q1))
Q10000 = c(o(Q1000, Q0))
Q10001 = c(o(Q1000, Q1))
Q10010 = c(o(Q1001, Q0))
Q10011 = c(o(Q1001, Q1))
Q10100 = c(o(Q1010, Q0))
Q10101 = c(o(Q1010, Q1))
Q10110 = c(o(Q1011, Q0))
Q10111 = c(o(Q1011, Q1))
Q11000 = c(o(Q1100, Q0))
Q11001 = c(o(Q1100, Q1))
Q11010 = c(o(Q1101, Q0))
Q11011 = c(o(Q1101, Q1))
Q11100 = c(o(Q1110, Q0))
Q11101 = c(o(Q1110, Q1))
Q11110 = c(o(Q1111, Q0))
Q11111 = c(o(Q1111, Q1))

## Identity matrices
I2 = np.identity(2)
I4 = np.identity(2**2)
I8 = np.identity(2**3)
I16 = np.identity(2**4)
I32 = np.identity(2**5)

# Define Gate operations
def PauliX(n): return np.array(([0., 1.], [1., 0.])).dot(n)
def PauliY(n): return np.array(([0., -1.0j], [1.0j, 0.])).dot(n)
def PauliZ(n): return np.array(([1., 0.], [0., -1.])).dot(n)

def Dagger(n): return np.conj(n).T

def Hadamard(n): return np.array(([1., 1.], [1., -1.])).dot(n)/np.sqrt(2)
def Hadamard4(n):
        """ This represents a 4X4 Hadamard Matrix, n is a 2-Qubit system """
        h=Hadamard(I2)
        x=c([h, h], axis=1)
        y=c([h, -h], axis=1)
        h4=c([x, y])
        return h4.dot(n)
def Hadamard8(n):
        """ This represents a 8X8 Hadamard Matrix, n is a 3-Qubit system """
        h4=Hadamard4(I4)
        x=c([h4, h4], axis=1)
        y=c([h4, -h4], axis=1)
        h8=c([x, y])
        return h8.dot(n)
def Hadamard16(n):
        """ This represents a 16X16 Hadamard Matrix, n is a 4-Qubit system """
        h8=Hadamard8(I8)
        x=c([h8, h8], axis=1)
        y=c([h8, -h8], axis=1)
        h16=c([x, y])
        return h16.dot(n)
def Hadamard32(n):
        """ This represents a 32X32 Hadamard Matrix, n is a 5-Qubit system """
        h16=Hadamard16(I16)
        x=c([h16, h16], axis=1)
        y=c([h16, -h16], axis=1)
        h32=c([x, y])
        return h32.dot(n)

def CNOT(n):
        x=np.copy(I4)
        t=np.copy(x[2,])
        x[2,]=x[3,]
        x[3,]=t
        return x.dot(n)

def PauliX_4(n):
        """ This represents the 4X4 pauliX matrix, with n as a 2-qubit system"""
        p=PauliX(I2)
        x=c([p, I2], axis=1)
        y=c([I2, p], axis=1)
        p4=c([x, y])
        return p4.dot(n)
def PauliY_4(n):
        """ This represents the 4X4 pauliY matrix, with n as a 2-qubit system"""
        p=PauliY(I2)
        x=c([p, -1j*I2], axis=1)
        y=c([1j*I2, p], axis=1)
        p4=c([x, y])
        return p4.dot(n)
def PauliZ_4(n):
        """ This represents the 4X4 pauliZ matrix, with n as a 2-qubit system"""
        p=PauliZ(I2)
        x=c([2*p, I2-I2], axis=1)
        y=c([I2-I2, -2*p], axis=1)
        p4=c([x, y])
        return p4.dot(n)

def Rotate(n, t): return np.array(([np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)])).dot(n)

def Phase(n): return np.array(([1., 0.], [0., 1.0j])).dot(n)
def Phase1(n): return np.array(([1., 0.], [0., -1.0j])).dot(n)
def T(n): return np.array(([1., 0.], [0., np.exp(1.0j*np.pi/4)])).dot(n)
def T1(n): return np.array(([1., 0.], [0., np.exp(-1.0j*np.pi/4)])).dot(n)

def Toffoli(n):
        """n must be a 8X8 matrix"""
        x=I8
        t=np.copy(x[6,])
        x[6,]=x[7,]
        x[7,]=t
        return x.dot(n)