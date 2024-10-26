import numpy as np
from typing import Dict
from enum import Enum

class Idx(Enum):
    U = 0  # contravariant
    D = 1  # covariant
    T = 2  # tetrad

class Kerr:
    """
    Kerr metric in Kerr-Schild coordinates.
    
    Methods
    -------
    z(x)
        Returns z = 2 r / Sigma
    alpha(x)
        Returns the time-lag factor
    betai(x)
        Returns the shift vector
    hij(x)
        Returns the 3-metric (upper indices)
    h_ij(x)
        Returns the 3-metric (lower indices)
    """

    def __init__(self, a):
        self._a = a
        
    @property
    def a(self):
        return self._a

    def Sigma(self, x):
        """
        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            Sigma = r^2 + a^2 cos^2 theta
        """
        return x[:, 0] ** 2 + self.a**2 * np.cos(x[:, 1]) ** 2

    def Delta(self, x):
        """
        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            Delta = r^2 - 2 r + a^2
        """
        return x[:, 0] ** 2 - 2 * x[:, 0] + self.a**2

    def A(self, x):
        """
        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            A = (r^2 + a^2)^2 - a^2 Delta sin^2 theta
        """
        return (x[:, 0] ** 2 + self.a**2) ** 2 - self.a**2 * self.Delta(x) * np.sin(
            x[:, 1]
        ) ** 2
    
    def z(self, x):
        """
        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            z = 2 r / Sigma
        """
        return 2 * x[:, 0] / self.Sigma(x)

    def alpha(self, x):
        """
        Computes time-lag factor

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            time-lag factor alpha
        """
        return np.sqrt(1 / (1 + self.z(x)))

    def betai(self, x):
        """
        Computes the shift vector

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3)
            shift vector beta^i
        """
        n = x.shape[0]
        return np.array([self.z(x) / (1 + self.z(x)), np.zeros(n), np.zeros(n)]).T

    def hij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric hij
        """
        n = x.shape[0]
        A = self.A(x)
        Sigma = self.Sigma(x)
        return np.array(
            [
                [A / (Sigma * (Sigma + 2 * x[:, 0])), np.zeros(n), self.a / Sigma],
                [np.zeros(n), 1 / self.Sigma(x), np.zeros(n)],
                [self.a / Sigma, np.zeros(n), 1 / (Sigma * np.sin(x[:, 1]) ** 2)],
            ]
        ).T

    def h_ij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric h_ij (Lower indices)
        """
        n = x.shape[0]
        A = self.A(x)
        z = self.z(x)
        Sigma = self.Sigma(x)
        a = self.a
        return np.array(
            [
                [1 + z, np.zeros(n), -a * (1 + z) * np.sin(x[:, 1]) ** 2],
                [np.zeros(n), Sigma, np.zeros(n)],
                [
                    -a * (1 + z) * np.sin(x[:, 1]) ** 2,
                    np.zeros(n),
                    A * np.sin(x[:, 1]) ** 2 / Sigma,
                ],
            ]
        ).T
    
class Minkowski:
    """
    Minkowski metric in Cartesian spherical coordinates.

    Methods
    -------
    alpha(x)
        Returns the time-lag factor
    betai(x)
        Returns the shift vector
    hij(x)
        Returns the 3-metric (upper indices)
    h_ij(x)
        Returns the 3-metric (lower indices)
    """
    def __init__(self): ...
        
    def alpha(self, x):
        """
        Computes time-lag factor
        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            time-lag factor alpha
        """
        return np.ones(x.shape[0])

    def betai(self, x):
        """
        Computes the shift vector

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3)
            shift vector beta^i
        """
        n = x.shape[0]
        return np.zeros((n, 3))

    def hij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric hij
        """
        n = x.shape[0]
        return np.array(
            [
                [np.ones(n), np.zeros(n), np.zeros(n)],
                [np.zeros(n), 1 / x[:, 0] ** 2, np.zeros(n)],
                [np.zeros(n), np.zeros(n), 1 / (x[:, 0] ** 2 * np.sin(x[:, 1]) ** 2)],
            ]
        ).T

    def h_ij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric hij
        """
        n = x.shape[0]
        return np.array(
            [
                [np.ones(n), np.zeros(n), np.zeros(n)],
                [np.zeros(n), x[:, 0] ** 2, np.zeros(n)],
                [np.zeros(n), np.zeros(n), x[:, 0] ** 2 * np.sin(x[:, 1]) ** 2],
            ]
        ).T
    

class grTools:

    def __init__(self, metric: str, metric_params: Dict[str, float] = None) -> None:
        assert metric in ['mink', 'kerr'], "Invalid metric type"
        if metric == 'mink':
            self.metric = Minkowski()
        if metric == 'kerr':
            self.metric = Kerr(metric_params.get("a"))

    @staticmethod
    def eLC_ijk(n):
        """
        Levi-Civita psuedotensor

        Parameters
        ----------
        n : int
            number of particles

        Returns
        -------
        np.array (n x 3 x 3 x 3)
            Levi-Civita psuedotensor
        """
        return -np.array(
            [
                [
                    [np.zeros(n), np.zeros(n), np.zeros(n)],
                    [np.zeros(n), np.zeros(n), np.ones(n)],
                    [np.zeros(n), -np.ones(n), np.zeros(n)],
                ],
                [
                    [np.zeros(n), np.zeros(n), -np.ones(n)],
                    [np.zeros(n), np.zeros(n), np.zeros(n)],
                    [np.ones(n), np.zeros(n), np.zeros(n)],
                ],
                [
                    [np.zeros(n), np.ones(n), np.zeros(n)],
                    [-np.ones(n), np.zeros(n), np.zeros(n)],
                    [np.zeros(n), np.zeros(n), np.zeros(n)],
                ],
            ]
        ).T
    
    
    def transform(self, v: np.ndarray[np.ndarray[float]], x: np.ndarray[np.ndarray[float]], frm: int, to: int) -> np.ndarray[np.ndarray[float]]:
        
        metric = self.metric
        
        if frm == Idx.T or to == Idx.T:
            n = x.shape[0]

            # defining metric components
            hrr = metric.hij(x)[:, 0, 0]
            h_tt = metric.h_ij(x)[:, 1, 1]
            h_pp = metric.h_ij(x)[:, 2, 2]
            h_rp = metric.h_ij(x)[:, 0, 2]

        # tetrad matrices
        if (frm == Idx.D and to == Idx.T) or (frm == Idx.T and to == Idx.U):
            ei_ih = np.array(
                [
                    [np.sqrt(hrr), np.zeros(n), np.zeros(n)],
                    [np.zeros(n), 1 / np.sqrt(h_tt), np.zeros(n)],
                    [-np.sqrt(hrr) * h_rp / h_pp, np.zeros(n), 1 / np.sqrt(h_pp)],
                ]
            ).T
        elif (frm == Idx.T and to == Idx.D) or (frm == Idx.U and to == Idx.T):
            eih_i = np.array(
                [
                    [1 / np.sqrt(hrr), np.zeros(n), h_rp / np.sqrt(h_pp)],
                    [np.zeros(n), np.sqrt(h_tt), np.zeros(n)],
                    [np.zeros(n), np.zeros(n), np.sqrt(h_pp)],
                ]
            ).T

        if frm == Idx.D and to == Idx.T:
            return np.einsum("nji,nj->ni", ei_ih, v)
        elif frm == Idx.T and to == Idx.D:
            return np.einsum("nij,nj->ni", eih_i, v)
        elif frm == Idx.U and to == Idx.T:
            return np.einsum("nji,nj->ni", eih_i, v)
        elif frm == Idx.T and to == Idx.U:
            #print(ei_ih, v)
            return np.einsum("nij,nj->ni", ei_ih, v)
        elif frm == Idx.D and to == Idx.U:
            return np.einsum("nij,nj->ni", metric.hij(x), v)
        elif frm == Idx.U and to == Idx.D:
            return np.einsum("nij,nj->ni", metric.h_ij(x), v)
        else:
            raise ValueError("Invalid transformation")