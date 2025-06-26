# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as cnp

cnp.import_array()

ctypedef cnp.float32_t  FLOAT32
ctypedef cnp.uint8_t    UINT8
ctypedef cnp.uint16_t   UINT16

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

cpdef tuple compute_means(cnp.ndarray[UINT16, ndim=3] image,
                          cnp.ndarray[UINT8,  ndim=3] u,
                          cnp.ndarray[UINT8,  ndim=3] mask):
    """Compute mean intensity outside/inside the *mask*."""
    cdef cnp.ndarray[UINT8, ndim=3] not_u = np.logical_not(u).astype(np.uint8)
    cdef cnp.ndarray[UINT8, ndim=3] u_masked = u * mask
    cdef cnp.ndarray[UINT8, ndim=3] not_u_masked = not_u * mask

    cdef cnp.ndarray[FLOAT32, ndim=3] image_f = image.astype(np.float32)

    cdef float c0 = (image_f * not_u_masked).sum() / float(not_u_masked.sum() + 1e-8)
    cdef float c1 = (image_f * u_masked).sum()     / float(u_masked.sum()     + 1e-8)
    return c0, c1


cpdef cnp.ndarray[FLOAT32, ndim=3] compute_force(
        cnp.ndarray[FLOAT32, ndim=3] abs_du,
        cnp.ndarray[UINT16,  ndim=3] image,
        float c0, float c1,
        float lambda1, float lambda2):

    cdef cnp.ndarray[FLOAT32, ndim=3] image_f = image.astype(np.float32)
    cdef cnp.ndarray[FLOAT32, ndim=3] result = abs_du * (
        lambda1 * (image_f - np.float32(c1))**2 -
        lambda2 * (image_f - np.float32(c0))**2
    )
    return result


cpdef void update_curve(cnp.ndarray[UINT8,   ndim=3] u,
                        cnp.ndarray[FLOAT32, ndim=3] aux,
                        cnp.ndarray[UINT8,   ndim=3] mask):
    """Binary threshold update of the level‑set function."""
    u[aux < 0] = 1
    u[aux > 0] = 0
    u *= mask

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

cpdef cnp.ndarray[UINT8, ndim=3] morphological_chan_vese(
        cnp.ndarray[UINT16, ndim=3] image,
        cnp.ndarray[UINT8,  ndim=3] mask,
        Py_ssize_t[:] shape,
        Py_ssize_t num_iter=15,
        float lambda1=1,
        float lambda2=1):
    """Segment *image* using MorphACWE."""

    cdef cnp.ndarray[cnp.int8_t, ndim=3] init_ls = checkerboard_level_set(shape)
    cdef cnp.ndarray[UINT8, ndim=3] u = (init_ls > 0).astype(np.uint8)

    cdef Py_ssize_t i
    cdef float c0, c1
    cdef cnp.ndarray[FLOAT32, ndim=3] abs_du, aux

    for i in range(num_iter):
        c0, c1 = compute_means(image, u, mask)

        abs_du = np.zeros_like(u, dtype=np.float32)
        for g in np.gradient(u):
            abs_du += np.abs(g.astype(np.float32))

        aux = compute_force(abs_du, image, c0, c1, lambda1, lambda2)
        update_curve(u, aux, mask)

    return u