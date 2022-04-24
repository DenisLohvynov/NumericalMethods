from collections.abc import Sequence, MutableSequence
from functools import reduce
from typing import Callable


def Jacobi(
        A: Sequence[Sequence[float]],
        b: Sequence[float],
        eps: float = 0.001,
        x_init: MutableSequence[float] | None = None) -> list[float]:
    """
    Jacobi method for A*x = b (*)

    :param A: Matrix (LHS)

    :param b: known vector (RHS)

    :param x_init: first guess

    :return: approximate solution to equation (*)
    """
    if not all((len(A) == len(A[i]) for i in range(len(A)))):
        raise ValueError('Wrong dimensions of the matrix A.')
    if len(b) != len(A) or ((x_init is not None) and
                            (len(b) != len(x_init) or len(A) != len(x_init))):
        raise ValueError('Wrong length of b or x or dimensions of A')

    for i in range(len(A)):
        if A[i][i] == 0:
            raise ZeroDivisionError(f'A[{i}][{i}] = 0!')

    # норма матрицы B (oo)
    q: float = max((reduce(lambda x, y: abs(x) + abs(y), a)/abs(A[i][i])-1
                   for i, a in enumerate(A)))
    if q >= 1:
        raise ValueError(f'||B|| which is -D**-1(L+R) > 1, namely {q}')

    def sum(a: Sequence[float], x: Sequence[float], j: int) -> float:
        S: float = 0
        for i, (m, y) in enumerate(zip(a, x)):
            if i != j:
                S += m*y
        return S

    def norm(x: Sequence[float], y: Sequence[float]) -> float:
        """
        norm of R^n l(oo)

        :param x: first vector

        :param y: second vector

        :return: norm of two vectors

        PS if WLOG len(x) > len(y) then x will be dealt with like x[:len(y)]
        """
        return max((abs(x0-y0) for x0, y0 in zip(x, y)))

    def check(q: float) -> Callable[[Sequence[float], Sequence[float]], float]:
        """
        :param q: norm of B

        :return: function to check the while loop
        """
        if q <= 0.5:
            return norm
        return lambda x, y: norm(x, y)*q/(1-q)

    if x_init is None:
        x_init = [a/A[i][i] for i, a in enumerate(b)]
    x: list[float] = [-(sum(a, x_init, i) - b[i])/A[i][i]
                      for i, a in enumerate(A)]
    func_check = check(q)
    while func_check(x_init, x) > eps:
        for i, elem in enumerate(x):
            x_init[i] = elem
        for i, a in enumerate(A):
            x[i] = -(sum(a, x_init, i) - b[i])/A[i][i]
    return x


if __name__ == '__main__':
    A: list[list[float]] = [
        [2.998, 0.209, 0.315, 0.281],
        [0.163, 3.237, 0.226, 0.307],
        [0.416, 0.175, 3.239, 0.159],
        [0.287, 0.196, 0.325, 4.062]
    ]
    b = [0.108, 0.426, 0.31, 0.84]
    print(Jacobi(A, b, 10**-5))
