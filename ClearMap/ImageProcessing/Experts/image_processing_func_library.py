from typing import Any, Dict, Callable

import numpy as np
import scipy.ndimage as _ndi
import skimage.filters as skif

import ClearMap.IO.IO as cmp_io
from ClearMap.ImageProcessing.Experts.block_function import block_function
import ClearMap.ImageProcessing.Filter.Rank as _rnk
import ClearMap.ImageProcessing.LightsheetCorrection as _lc
import ClearMap.ImageProcessing.LocalStatistics as ls
import ClearMap.ImageProcessing.Differentiation.Hessian as hes


ImageProcessingFunction = Callable[..., Dict[str, Any]]


@block_function
def clip(source, *, clip_range=(300, 60000), max_bin=None, **kwargs) -> Dict[str, np.ndarray]:
    """
        Clip and normalize array values to a specified range.

        Parameters
        ----------
        source : array_like
            Source data to be clipped and normalized.
        clip_range : tuple of (float, float)
            Lower and upper clipping bounds (clip_low, clip_high).
        max_bin : int, optional
            Maximum value for normalization. Output range is [0, norm-1].
        dtype : data-type, optional
            Data type of the output clipped array.

        Returns
        -------
        dict of {str : np.ndarray}
            clipped : np.ndarray
                Clipped and normalized array.
            mask : np.ndarray
                Boolean mask indicating values within the clip range.
            high_mask : np.ndarray
                Boolean mask indicating values that were clipped high (>= clip_high).
            not_low_mask : np.ndarray
                Boolean mask indicating values that were not clipped low (>= clip_low).
    """
    clip_low, clip_high = clip_range

    clipped = np.array(source[:], dtype=float)

    low_mask = clipped < clip_low
    clipped[low_mask] = clip_low

    high_mask = clipped >= clip_high
    clipped[high_mask] = clip_high

    mask = np.logical_not(np.logical_or(low_mask, high_mask))
    clipped -= clip_low
    clipped *= float(max_bin - 1) / (clip_high - clip_low)

    return {'clipped': clipped,
            'mask': mask,
            'high_mask': high_mask,
            'not_low_mask': ~low_mask}


@block_function
def lightsheet_correction(source, mask, **params) -> Dict[str, np.ndarray]:
    # TODO the ideal would be to directly wrap _lc.correct_lightsheet as a block function
    lc_corrected = _lc.correct_lightsheet(source, mask=mask, **params)
    return {'lc_corrected': lc_corrected}


@block_function
def median(source, not_low_mask, **params) -> Dict[str, np.ndarray]:
    # TODO the ideal would be to directly wrap _rnk.median as a block function
    median = _rnk.median(source, mask=not_low_mask,**params)
    return {'median': median}


@block_function
def deconvolve(source, binarized_mask, *, sigma, **kwargs) -> Dict[str, np.ndarray]:
    """
    Remove halos by deconvolving Gaussian-filtered signal from masked regions.

    Creates a smoothed background estimate from masked regions using Gaussian
    filtering, then subtracts it from the source while preserving the original
    masked values.

    Args:
        source (np.ndarray): Input 3D array to deconvolve.
        binarized_mask (np.ndarray): Boolean mask indicating regions to preserve.
        sigma (float): The std of a Gaussian filter applied to the high intensity pixel image.
                       The number should reflect the scale of the halo effect seen around high
                       intensity structures. For the vasculature a typical value is 10.

    Returns:
        dict: Dictionary containing:
            - 'deconvolved' (np.ndarray): Halo-subtracted array with
              original values preserved in masked regions.
    """
    binarized = source[binarized_mask] # WARNING copy to prevent multiprocessing race condition

    convolved = np.zeros(source.shape, dtype=float)
    convolved[binarized_mask] = binarized

    for z in range(convolved.shape[2]):
        convolved[:, :, z] = _ndi.gaussian_filter(convolved[:, :, z], sigma=sigma)

    deconvolved = source - np.minimum(source, convolved)
    deconvolved[binarized_mask] = binarized
    return {'deconvolved': deconvolved}


@block_function
def threshold_bin(source, *, threshold=None, **kwargs) -> Dict[str, np.ndarray]:
    """
    Create a binary mask by applying a simple threshold to the source array.

    Parameters
    ----------
    source : np.ndarray
        Input array to threshold.
    threshold : float, optional
        Threshold value. If None, returns an array of zeros.

    Returns
    -------
    dict of {str : np.ndarray}
        binary : np.ndarray
            Boolean array where True indicates values above threshold.
    """
    if threshold:
        binary = source > threshold
    else:
        binary = np.zeros_like(source)
    return {'binary': binary}


@block_function
def adjust_gamma(source, *, gamma=0.45, gain=1, **kwargs) -> Dict[str, np.ndarray]:
    """
    Apply gamma correction to adjust image contrast.

    Performs power-law transformation: out = ((source / scale)^gamma) * scale * gain

    Parameters
    ----------
    source : np.ndarray
        Input array to adjust.
    gamma : float, optional
        Gamma value for power-law transformation. Must be non-negative.
        Values < 1 brighten the image, values > 1 darken it. Default is 0.45.
    gain : float, optional
        Multiplicative factor applied after gamma correction. Default is 1.

    Returns
    -------
    dict of {str : np.ndarray}
        gamma_adjusted : np.ndarray
            Gamma-corrected array.

    Raises
    ------
    ValueError
        If gamma is negative.
    """
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")
    elif gamma == 1:
        return source
    dtype = source.dtype.type
    scale = float(np.max(source) - np.min(source))
    out = (((source / scale) ** gamma) * scale * gain).astype(dtype)

    return {'gamma_adjusted': out}


def threshold_isodata(source, **kwargs) -> int:
    """
    Calculate threshold using the ISODATA algorithm.

    Wrapper around scikit-image's ISODATA thresholding method.
    Returns the highest threshold if multiple values are found, or 1 if the algorithm fails.

    Parameters
    ----------
    source : np.ndarray
        Input array for threshold calculation.

    Returns
    -------
    float
        Calculated threshold value, or 1 if calculation fails.
    """
    try:
        thresholds = skif.threshold_isodata(source, return_all=True)
        if len(thresholds) > 0:
            return thresholds[-1]
        else:
            return 1
    except:  # FIXME: too broad
        return 1


@block_function
def threshold_adaptive(source, mask=None, *, function=threshold_isodata, selem=(100, 100, 3), spacing=(25, 25, 3),
                       interpolate=1, step=None, **kwargs) -> Dict[str, Any]:
    """
        Apply adaptive thresholding using local threshold calculations.

        Divides the image into blocks and applies a threshold function locally,
        then interpolates to create a spatially-varying threshold map.

        Parameters
        ----------
        source : array_like
            Input array or source object to threshold.
        mask : np.ndarray, optional
            Mask to restrict threshold calculation regions.
        function : callable, optional
            Function to calculate local thresholds. Default is threshold_isodata.
        selem : tuple of int, optional
            The structural element size to estimate the percentiles.
            Should be larger than the larger vessels.
            For the vasculature a typical value is (200,200,5).
        spacing : tuple of int, optional
            The spacing used to move the structural elements.
            Larger spacings speed up processing but become locally less precise.
            For the vasculature a typical value is (50,50,5)
        interpolate : int, optional
            The order of the interpolation used in constructing the full
            background estimate in case a non-trivial spacing is used.
            For the vasculature a typical value is 1.
        step : tuple of int or None
            If tuple, subsample the local region by these step. Note that the
            structural_element is applied after this subsampling.

        Returns
        -------
        dict of {str : np.ndarray}
            bin_adaptive : np.ndarray
                Boolean array from adaptive thresholding.
            threshold : np.ndarray
                Spatially-varying threshold map.
        """
    source = cmp_io.as_source(source)[:]
    threshold = ls.apply_local_function(source, function=function, mask=mask, dtype=float,
                                        selem=selem, spacing=spacing, interpolate=interpolate, step=step)
    binary = source > threshold
    return {'bin_adaptive': binary, 'threshold': threshold}


@block_function
def equalize(source, mask=None, *, percentile=(0.5, 0.95), max_value=1.5, selem=(200, 200, 5), spacing=(50, 50, 5),
             interpolate=1, **kwargs) -> Dict[str, np.ndarray]:
    """
    Perform local histogram equalization based on percentile normalization.

    Normalizes the image locally using percentile values to correct for
    illumination variations while preventing over-amplification.

    See Also
    --------
    ClearMap.ImageProcessing.LocalStatistics.local_percentile

    Parameters
    ----------
    source : np.ndarray
        Input array to equalize.
    mask : np.ndarray, optional
        Mask indicating regions to process.
    percentile : tuple of float, optional
        The (lower, upper) percentile values used to estimate the equalization.
        The lower percentile is used for normalization, the upper to limit the
        maximal boost to a maximal intensity above this percentile.
        For vasculature, a typical value is (0.4, 0.975).
    max_value : float, optional
        The maximal intensity value in the equalized image. Maximum allowed
        normalization factor to prevent over-amplification.
        For vasculature, a typical value is 1.5.
    selem : tuple of int, optional
        The structural element size to estimate the percentiles.
        Should be larger than the largest vessels.
        For vasculature, a typical value is (200, 200, 5).
    spacing : tuple of int, optional
        The spacing used to move the structural elements.
        Larger spacings speed up processing but become locally less precise.
        For vasculature, a typical value is (50, 50, 5).
    interpolate : int, optional
        The order of the interpolation used in constructing the full
        background estimate in case a non-trivial spacing is used.
        For vasculature, a typical value is 1.

    Returns
    -------
    dict of {str : np.ndarray}
        equalized : np.ndarray
            Locally normalized array
    """
    equalized = ls.local_percentile(source, percentile=percentile, mask=mask, dtype=float,
                                    selem=selem, spacing=spacing, interpolate=interpolate)
    normalize = 1 / np.maximum(equalized[..., 0], 1)
    maxima = equalized[..., 1]
    ids = maxima * normalize > max_value
    normalize[ids] = max_value / maxima[ids]
    equalized = np.array(source, dtype=float) * normalize
    return {'equalized': equalized}


@block_function
def bin_equalized(source, *, threshold, max_bin=None, **kwargs) -> Dict[str, np.ndarray]:
    """
        Threshold and scale source array based on background level.

        Creates a binary mask and scales the background (non-thresholded) regions
        to the full dynamic range.

        Parameters
        ----------
        source : np.ndarray
            Input array to process.
        threshold : float
            Voxels above this threshold will be added to the binarization result
            in the multi-path binarization.
            For the vasculature a typical value is 1.1.
        max_bin : int
            Number of intensity levels to use for the data after preprocessing.
            Higher values will increase the intensity resolution but slow down
            processing.
            For the vasculature a typical value is 2**12.

        Returns
        -------
        dict of {str : np.ndarray}
            bg_scaled_equalized : np.ndarray
                Background-scaled array with values above threshold capped.
            bin_equalized : np.ndarray
                Boolean mask of values above threshold.
        """
    bin_equalized = source > threshold
    source[bin_equalized] = threshold
    bg_scaled_equalized = float(max_bin - 1) / threshold * source
    return {'bg_scaled_equalized': bg_scaled_equalized, 'bin_equalized': bin_equalized}


def tubify(source, sigma=1.0, gamma12=1.0, gamma23=1.0, alpha=0.25) -> np.ndarray:
    return hes.lambda123(source=source, sink=None, sigma=sigma, gamma12=gamma12, gamma23=gamma23, alpha=alpha)


@block_function
def vesselize(source, mask, *, background_params, tubeness=None, threshold=None,
              max_bin=None, **kwargs) -> Dict[str, np.ndarray]:
    """
    Detect vessel-like structures.

    Combine optional background subtraction, Hessian-based tubeness detection,
    and thresholding to identify vessel structures.

    Parameters
    ----------
    source : np.ndarray
        Input array for vessel detection.
    mask : np.ndarray
        Mask indicating regions to process.
    background_params : dict or None
        Parameters to correct for local background. See
        :func:`ClearMap.ImageProcessing.Filter.Rank.percentile`.
        If None, no background correction is done before the tube filter.

        selem : tuple
            The structural element specification to estimate the percentiles.
            Should be larger than the largest vessels intended to be
            boosted by the tube filter.
            For the vasculature a typical value is ('disk', (30,30,1)).

        percentile : float
            Percentile in [0,1] used to estimate the background.
            For the vasculature a typical value is 0.5.

    tubeness : dict
        Parameters used for the tube filter. See
        :func:`ClearMap.ImageProcessing.Differentiation.Hessian.lambda123`.

        sigma : float
            The scale of the vessels to boos in the filter.
            For the vasculature a typical value is 1.0.

    threshold : float
        Voxels above this threshold will be added to the binarization result
        in the multi-path binarization.
        For the vasculature a typical value is 120.
    max_bin : int, optional
        Number of intensity levels to use for the data after preprocessing.
        Higher values will increase the intensity resolution but slow down
        processing.
        For the vasculature a typical value is 2**12.

    Returns
    -------
    dict of {str : np.ndarray}
        tubeness : np.ndarray
            Continuous tubeness measure.
        bin_vesselized : np.ndarray
            Binary mask of detected vessels.
    """
    if background_params:
        source_reduced = np.array(source, dtype='uint16')  # TODO data types should be specified outside of core functions
        bg = _rnk.percentile(source_reduced, max_bin=max_bin, mask=mask, **{**background_params})
        bg_subtracted = source_reduced - np.minimum(source_reduced, bg)
    else:
        bg_subtracted = source

    tubeness_par = tubeness if tubeness else {}
    tubeness = tubify(bg_subtracted, **tubeness_par)

    bin_vesselized = np.zeros(source.shape, dtype=np.uint8)
    if threshold:
        bin_vesselized = tubeness > threshold

    return {'tubeness': tubeness, 'bin_vesselized': bin_vesselized}