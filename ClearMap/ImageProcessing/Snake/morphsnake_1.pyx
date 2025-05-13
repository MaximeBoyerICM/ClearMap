# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float32_t FLOAT32
ctypedef cnp.uint8_t   UINT8

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

cpdef checkerboard_level_set(shape, int square_size=15):
    """Return a binary checkerboard level‑set the same shape as *shape*."""
    grid = np.mgrid[[slice(i) for i in shape]]
    grid = grid // square_size
    grid &= 1
    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    return np.int8(checkerboard)

# ----------------------------------------------------------------------------
# Core primitives
# ----------------------------------------------------------------------------

cpdef tuple compute_means(cnp.ndarray[FLOAT32, ndim=3] image,
                          cnp.ndarray[UINT8,   ndim=3] mask):
    """Compute mean intensity outside/inside the *mask*."""
    cdef cnp.ndarray[UINT8, ndim=3] not_mask = np.logical_not(mask).astype(np.uint8)
    cdef float c0 = (image * not_mask).sum() / float(not_mask.sum() + 1e-8)
    cdef float c1 = (image * mask).sum()     / float(mask.sum()     + 1e-8)
    return c0, c1


cpdef cnp.ndarray[cnp.float64_t, ndim=3] compute_force(
        cnp.ndarray[cnp.float64_t, ndim=3] abs_du,
        cnp.ndarray[FLOAT32, ndim=3] image,
        float c0, float c1,
        float lambda1, float lambda2):
    """Return the geodesic force term."""
    return abs_du * (
        lambda1 * (image - c1)**2 -
        lambda2 * (image - c0)**2
    )


cpdef void update_curve(cnp.ndarray[UINT8,   ndim=3] u,
                        cnp.ndarray[cnp.float64_t, ndim=3] aux):
    """Binary threshold update of the level‑set function."""
    u[aux < 0] = 1
    u[aux > 0] = 0


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

cpdef cnp.ndarray[UINT8, ndim=3] morphological_chan_vese(
        cnp.ndarray[FLOAT32, ndim=3] image,
        Py_ssize_t[:] shape,
        Py_ssize_t num_iter=15,
        float lambda1 = 1,
        float lambda2 = 1):
    """Segment *image* using MorphACWE.

    Parameters
    ----------
    image : ndarray[float32] (Z, Y, X)
        3‑D scalar field to segment. Values should be scaled 0‑1.
    num_iter : int
        Number of iterations.
    lambda1, lambda2 : float
        Chan–Vese region weight parameters.
    """

    # Initial level‑set: checkerboard inside a contiguous uint8 mask
    cdef cnp.ndarray[cnp.int8_t, ndim=3] init_ls = checkerboard_level_set(shape)
    cdef cnp.ndarray[UINT8, ndim=3] u = (init_ls > 0).astype(np.uint8)

    cdef Py_ssize_t i
    cdef float c0, c1
    cdef cnp.ndarray[cnp.float64_t, ndim=3] du, abs_du, aux

    for i in range(num_iter):
        # region means
        c0, c1 = compute_means(image, u)

        # |∇u|
        abs_du = np.zeros_like(u, dtype="float64")
        for g in np.gradient(u):
            abs_du += np.abs(g)

        # update
        aux = compute_force(abs_du, image, c0, c1, lambda1, lambda2)
        update_curve(u, aux)

    return u
