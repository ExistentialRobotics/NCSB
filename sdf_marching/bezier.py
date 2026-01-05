import numpy as np
from scipy.special import binom, beta
import cvxpy

class BezierPolynomial:
    def __init__(self, points, duration=1.0, _parent=None):
        self.points = points
        self._parent = _parent
        self._duration = duration

    #TODO: account for case when t is not numpy array
    def coefficient(self, t, i):
        binom_coefficient = binom(self.order, i)
        t_power_i = np.power(t[..., np.newaxis], i)
        one_m_t_power_n_m_i = np.power(1.0 - t[..., np.newaxis], self.order - i)

        return binom_coefficient * t_power_i * one_m_t_power_n_m_i

    def coefficient_matrix(self, t):
        return self.coefficient(t, np.arange(self.order+1))

    def query(self, t):
        return self.coefficient_matrix(t) @ self.points

    # return another BezierPolynomial object    
    def derivative(self):
        return BezierPolynomial(
            self.order * (self.points[1:, ...] - self.points[:-1, ...])
        )

    def integral(self):
        pass

    def norm_square_integral(self):
        coeff_matrix = sum(
            build_coeff_mat(self.order, self.order, k)
            for k in range(2 * self.order + 1)
        ) # sum over k so that positive definiteness becomes clearer - otherwise cvxpy can't track
        return sum(
            cvxpy.quad_form(
                self.points[..., d], coeff_matrix
            ) # use quad_form so that cvxpy can keep track of convexity
            for d in range(self.ndim)
        ) / (self.order + 1)

    #TODO later... 
    # def mul(self, other):
    #     """Computes the product of two Bezier curves.

    #     Paper Reference: Property 7: Arithmetic Operations

    #     Source: Section 5.1 of "The Bernstein Polynomial Basis: A Centennial
    #     Retrospective" by Farouki.

    #     :param multiplicand: Multiplicand
    #     :type multiplicand: Bezier
    #     :return: Product of the two curve
    #     :rtype: Bezier
    #     """
    #     if not isinstance(other, BezierPolynomial):
    #         msg = 'The multiplicand must be a {} object, not a {}'.format(
    #                 BezierPolynomial, type(other))
    #         raise TypeError(msg)

    #     dim = self.ndim
    #     if other.ndim != dim:
    #         msg = ('The dimension of both Bezier curves must be the same.\n'
    #                'The first dimension is {} and the second is {}'.format(
    #                        dim, other.dim))
    #         raise ValueError(msg)


    #     m = self.order
    #     n = other.order

    #     product = [
    #         [
    #             self.points[..., d].T @ build_coeff_mat(m, n, k) @ other.points[..., d]
    #             for d in range(dim)
    #         ]
    #         for k in range(m+n+1)
    #     ]

    #     return product

    @property
    def start(self):
        return self.points[0, :]

    @property
    def end(self):
        return self.points[-1, :]

    @property
    def order(self):
        return self.points.shape[0] - 1

    @property
    def is_n_dimensional(self):
        return self.points.ndim > 1

    @property
    def ndim(self):
        return self.points.shape[-1] if self.is_n_dimensional else 1

    @property
    def duration(self):
        traced_parent = self
        while traced_parent.parent is not None:
            traced_parent = traced_parent.parent
        return traced_parent.duration

    def __call__(self, t):
        return self.query(t)

# TODO:
#    Change this function name to prodM.
#    Clean up and slightly change _normSquare to accommodate this change
# def product_coefficient(m, n=None):
#     """Produces a product matrix for obtaining the product of two Bezier curves

#     This function computes the matrix of coefficients for multiplying two
#     Bezier curves. This function exists so that the coefficients matrix can be
#     computed ahead of time when performing many multiplications.

#     :param m: Degree of the first Bezier curve
#     :type m: int
#     :param n: Degree of the second Bezier curve
#     :type n: int
#     :return: Product matrix
#     :rtype: numpy.ndarray
#     """

#     if n is None:
#         n = m

#     coefMat = np.zeros(((m+1)*(n+1), m+n+1))

#     for k in range(m+n+1):
#         den = binom(m+n, k)
#         for j in range(max(0, k-n), min(m, k)+1):
#             print(m * j + k)
#             coefMat[m*j+k, k] = binom(m, j)*binom(n, k-j)/den

#     return coefMat

# def multiply_points_1d(multiplier, multiplicand, coefMat=None):
#     """Multiplies two Bezier curves together

#     The product of two Bezier curves can be computed directly from their
#     control points. This function specifically uses matrix multiplication to
#     increase the speed of the multiplication.

#     Note that this function is made for 1D control points.

#     One can pass in a matrix of product coefficients to compute the product
#     about 10x faster. It is recommended that the coefficient matrix is
#     precomputed and saved in memory when performing multiplication many times.
#     The function bezProductCoefficients will produce this matrix.

#     :param multiplier: Control points of the multiplier curve. Single dimension
#     :type multiplier: numpy.ndarray
#     :param multiplicand: Control points of the multiplicand curve.
#     :type multiplicand: numpy.ndarray
#     :param coefMat: Precomputed coefficient matrix from bezProductCoefficients
#     :type coefMat: numpy.ndarray or None
#     :return: Product of two Bezier curves
#     :rtype: numpy.ndarray

#     """

#     # This is faster for numpy, but using a naive method for cvxpy compatibility..

#     if isinstance(multiplicand, cvxpy.Expression) or isinstance(multiplier, cvxpy.Expression):
#         augMat = cvxpy.outer(multiplicand, multiplier)
#         newMat = cvxpy.reshape(augMat, (1, augMat.size))
#     else:
#         augMat = np.atleast_2d(multiplier).T @ np.atleast_2d(multiplicand)
#         newMat = augMat.reshape((1, augMat.size))
    
#     m = multiplier.size - 1
#     n = multiplicand.size - 1

#     if coefMat is None:
#         coefMat = product_coefficient(m, n)

#     return newMat @ coefMat

# def multiply_1d_naive(multiplier, multiplicand):

#     m = multiplier.shape[1] - 1
#     n = multiplicand.shape[1] - 1

#     product = [
#         multiplier.T @ build_coeff_mat(m, n, k) @ multiplicand
#         for k in range(m+n+1)
#     ]

#     return product

def build_coeff_mat(m, n, k):

    matrix = np.zeros(
        (m+1, n+1)
    )
    den = binom(m+n, k)
    for j in range(
        max(0, k - n),
        min(m, k) + 1
    ):
        matrix[j, k-j] = binom(m, j)*binom(n, k-j)/den
    
    return matrix

# TODO: make this work at some point!
# def norm_square(x, Nveh, Ndim, prodM):
#     """Compute the control points of the square of the norm of a vector

#     normSquare(x, Nveh, Ndim, prodM)

#     INPUT: Ndim*Nveh by N matrix x = [x1,...,x_Nveh)], x_i in R^Ndim
#     OUTPUT: control points of ||x_i||^2 ... Nveh by N matrix

#     Code ported over from Venanzio Cichella's MATLAB norm_square function.
#     NOTE: This only works on 1D or 2D matricies. It will fail for 3 or more.
#     """
# #    x = np.array(x)
# #    if x.ndim == 1:
# #        x = x[None]

#     m, N = x.shape

#     xsquare = np.zeros((m, prodM.shape[0]))

#     for i in range(m):
# #        xaug = np.dot(x[i, None].T, x[i, None])
#         xaug = np.dot(x.T, x)
#         xnew = xaug.reshape((N**2, 1))
#         xsquare[i, :] = np.dot(prodM, xnew).T[0]

#     S = np.zeros((Nveh, Nveh*Ndim))

#     for i in range(Nveh):
#         for j in range(Ndim):
#             S[i, Ndim*i+j] = 1

#     return np.dot(S, xsquare)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 0.0]
        ]
    )

    bp = BezierPolynomial(points)


    points_cvx = cvxpy.Variable(points.shape)
    bpc = BezierPolynomial(points_cvx)
    points_cvx.value = points


    times = np.linspace(0., 1.0, 100)
    traj = bp.query(times)

    fig, ax = plt.subplots()

    ax.plot(points[:, 0], points[:, 1], 'rx')
    ax.plot(traj[:, 0], traj[:, 1], 'k')
    plt.show(block=False)



