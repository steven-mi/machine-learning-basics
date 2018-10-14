""" Some functions for basic linear algebra opertations based on python lists. """
import functools
import random


def vector_add(a, b):
    if len(a) == len(b):
        return [(A + B) for A, B in zip(a, b)]
    return None


def vector_sub(a, b):
    if len(a) == len(b):
        return [(A - B) for A, B in zip(a, b)]
    return None


def vector_scalar_mul(r, a):
    return [r * x for x in a]


def vector_dot(a, b):
    if len(a) == len(b):
        return sum([(A * B) for A, B in zip(a, b)])
    return None


def create_random_matrix(n, m):
    return [[random.randint(0, 255) for x in range(n)] for y in range(m)]


def matrix_vector_mul(mat, vec):
    # l = [0] * len(mat)
    # for i in range(len(mat)):
    #  for x, y in zip(mat[i], vec):
    #       l[i] = l[i] + x * y
    # return l
    # hard way above easy way below
    return [sum([(x * y) for x, y in zip(i, vec)]) for i in mat]


def matrix_transpose(a):
    return [i for i in zip(*a)]
