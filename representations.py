__author__ = 'fbuettner'
import numpy as np


class AbstractRepresentation(object):
    def rep(self, state):
        raise NotImplementedError

    def theta(self):
        raise NotImplementedError

    def _add_bias(self, rep, state):
        if len(state.shape) > 1:
            return np.concatenate((rep, np.ones((state.shape[0], 1))), axis=1)
        else:
            return np.append(rep, 1.0)


class LinearRepresentation(AbstractRepresentation):
    def rep(self, state):
        r = np.abs(state - 5)
        return self._add_bias(r, state)

    def theta(self):
        return np.zeros(3)


class SquareRepresentation(AbstractRepresentation):
    def rep(self, state):
        r = np.abs(state - 5) ** 2
        return self._add_bias(r, state)

    def theta(self):
        return np.zeros(3)

class EuclidRepresentation(AbstractRepresentation):
    def rep(self, state):
        # if len(state.shape) == 1:
        # else:
        if len(state.shape) > 1:
            r = np.atleast_2d(np.array([np.sqrt(np.sum((np.atleast_2d((state-5)**2)), axis=1))])).T
        else:
            r = np.array([np.sqrt(np.sum((np.atleast_2d((state - 5) ** 2)), axis=1))])
        return self._add_bias(r, state)

    def theta(self):
        return np.zeros(2)


class PolynomialRepresentation(AbstractRepresentation):
    def rep(self, state):
        try:
            s = np.sum(state - 5, axis=1)
            r = np.c_[np.abs(state - 5), np.abs(state - 5) ** 2, s]
        except ValueError:
            # if state is just a single state, not an array of states
            s = np.sum(state - 5)
            r = np.append(np.append(np.abs(state - 5), np.abs(state - 5) ** 2), s)
        return self._add_bias(r, state)

    def theta(self):
        return np.zeros(6)


if __name__ == '__main__':
    sr = PolynomialRepresentation()
    state = np.array([[0, 0], [1, 0], [4, 3], [5, 5]])
    print sr.rep(state)
