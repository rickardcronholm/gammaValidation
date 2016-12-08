import dicom
import numpy as np
import glob
import os
from scipy.ndimage import map_coordinates
from scipy import interpolate
from progressbar import ProgressBar


def interp3(y, x, z, v, yi, xi, zi, **kwargs):
    def index_coords(corner_locs, interp_locs):
        # get coords to interpolate at
        index = np.arange(len(corner_locs))
        if np.all(np.diff(corner_locs) < 0):
            corner_locs, index = corner_locs[::-1], index[::-1]
        f = interpolate.interp1d(corner_locs, index, fill_value="extrapolate")
        return(f(interp_locs))

    orig_shape = np.asarray(yi).shape
    yi, xi, zi = np.atleast_1d(yi, xi, zi)
    for arr in [yi, xi, zi]:
        arr.shape = -1

    output = np.empty(yi.shape, dtype=float)
    coords = [index_coords(*item) for item in zip([y, x, z], [yi, xi, zi])]

    # linear interpolation of v at coords
    map_coordinates(v, coords, order=1, output=output, **kwargs)
    return output.reshape(orig_shape)


def load_dose_from_dicom(dcm):
    """Imports the dose in matplotlib format, with the following index mapping:
        i = y
        j = x
        k = z

    Therefore when using this function to have the coords match the same order,
    ie. coords_reference = (y, x, z)
    """
    pixels = np.transpose(
        dcm.pixel_array, (1, 2, 0))
    dose = pixels * dcm.DoseGridScaling

    return dose


def load_xyz_from_dicom(dcm):
    """Although this coordinate pull from Dicom works in the scenarios tested
    this is not an official x, y, z pull. It needs further confirmation.
    """
    resolution = np.array(
        dcm.PixelSpacing).astype(float)
    # Does the first index match x?
    # Haven't tested with differing grid sizes in x and y directions.
    dx = resolution[0]

    # The use of dcm.Columns here is under question
    x = (
        dcm.ImagePositionPatient[0] +
        np.arange(0, dcm.Columns * dx, dx))

    # Does the second index match y?
    # Haven't tested with differing grid sizes in x and y directions.
    dy = resolution[1]

    # The use of dcm.Rows here is under question
    y = (
        dcm.ImagePositionPatient[1] +
        np.arange(0, dcm.Rows * dy, dy))

    # Is this correct?
    z = (
        np.array(dcm.GridFrameOffsetVector) +
        dcm.ImagePositionPatient[2])

    return x, y, z


def gamma_calc(dose_evl, y_evl, x_evl, z_evl, dose_ref, y_ref, x_ref, z_ref,
dose_crit, dist_crit, search_dist, sampling, spacing, searchBox):
    # set boundaries and number of elements in each direction
    y_start = y_ref[0]
    y_end = y_ref[-1]
    y_n = int(np.abs(y_start - y_end) / (spacing[0] / sampling)) + 1
    x_start = x_ref[0]
    x_end = x_ref[-1]
    x_n = int(np.abs(x_start - x_end) / (spacing[1] / sampling)) + 1
    z_start = z_ref[0]
    z_end = z_ref[-1]
    z_n = int(np.abs(z_start - z_end) / (spacing[2] / sampling)) + 1
    # genreate mgrid for interpolation points
    yi, xi, zi = np.mgrid[y_start:y_end:y_n * 1j,
    x_start:x_end:x_n * 1j,
    z_start:z_end:z_n * 1j]
    # interpolate dose at positions (use np.nan if outside boundaries)
    d_ref = interp3(y_ref, x_ref, z_ref, dose_ref, yi, xi, zi,
        mode='constant', cval=np.nan)
    d_ref = np.reshape(d_ref, -1)
    # compute euclidiean distance to all interpolated points
    distance = np.power(np.power(yi - y_evl, 2) + np.power(xi - x_evl, 2) +
    np.power(zi - z_evl, 2), .5)
    # remove points where d_ref = np.nan
    pop_dose = np.where(d_ref == np.nan)
    d_ref = np.delete(d_ref, pop_dose)
    distance = np.delete(distance, pop_dose)
    # remove points where distance > search_dist
    pop_indx = np.where(distance > search_dist)
    distance = np.delete(distance, pop_indx)
    d_ref = np.delete(d_ref, pop_indx)
    # compute dose and dist part of gamma separately
    dose = np.power(d_ref - dose_evl, 2) / dose_crit ** 2
    dist = np.power(distance, 2) / dist_crit ** 2
    # find min of gamma and return it
    gamma = np.min(dose + dist)
    return np.sqrt(gamma)


def wrapper(directory, **kwargs):
    # locate and sort files
    files = glob.glob(os.path.join(directory, 'RD*dcm'))
    files.sort(key=os.path.getmtime)

    # extract data
    dcm_ref = dicom.read_file(files[0])  # oldest is reference
    dcm_evl = dicom.read_file(files[1])  # next is evaluation
    # extract doses and coordinate arrays
    dose_reference = load_dose_from_dicom(dcm_ref)
    dose_evaluation = load_dose_from_dicom(dcm_evl)
    x_reference, y_reference, z_reference = load_xyz_from_dicom(dcm_ref)
    x_evaluation, y_evaluation, z_evaluation = load_xyz_from_dicom(dcm_ref)

    # perform gamma computation
    gamma = compute_gamma(dose_reference, x_reference, y_reference, z_reference,
        dose_evaluation, x_evaluation, y_evaluation, z_evaluation, **kwargs)
    valid_gamma = gamma[~np.isnan(gamma)]
    print (('PassRate: ', np.sum(valid_gamma <= 1) / float(len(valid_gamma)) * 100))
    return gamma


def compute_gamma(dose_reference, x_reference, y_reference, z_reference,
        dose_evaluation, x_evaluation, y_evaluation, z_evaluation,
        dose_crit=3., dist_crit=3., sampling=2, cut_off=.2):
    # get absolute dose_crit
    dose_crit = dose_crit / 100.0 * np.max(dose_reference)
    # set cut off
    cut_val = cut_off * np.max(dose_reference)

    # find the length, in elements, to search
    dist_step = 2.0
    x_spacing = np.mean(np.unique(np.diff(x_reference)))
    y_spacing = np.mean(np.unique(np.diff(y_reference)))
    z_spacing = np.mean(np.unique(np.diff(z_reference)))
    spacing = [x_spacing, y_spacing, z_spacing]
    search_dist = 2 * dist_step
    search_X = int(np.ceil(search_dist / x_spacing))
    search_Y = int(np.ceil(search_dist / y_spacing))
    search_Z = int(np.ceil(search_dist / z_spacing))
    serchBox = [search_Y, search_X, search_Z]

    # start gamma computation
    pbar = ProgressBar()
    gamma = np.zeros(dose_evaluation.shape) * np.nan
    for u in pbar(list(range(0, len(y_evaluation)))):
        # find start, stop in y. start and stop may not extend outside volume
        yc = np.max([(np.abs(y_reference - y_evaluation[u])).argmin(), 0])
        yi_start = np.max([yc - search_Y, 0])
        yi_end = np.min([yc + search_Y, len(y_reference) - 1])
        for v in range(0, len(x_evaluation)):
            xc = np.max([(np.abs(x_reference - x_evaluation[v])).argmin(), 0])
            xi_start = np.max([xc - search_X, 0])
            xi_end = np.min([xc + search_X, len(x_reference) - 1])
            # find start, stop in x
            for w in range(0, len(z_evaluation)):
                zc = np.max([(np.abs(z_reference -
                z_evaluation[w])).argmin(), 0])
                zi_start = np.max([zc - search_Z, 0])
                zi_end = np.min([zc + search_Z, len(z_reference) - 1])
                # find start, stop in z
                if dose_evaluation[u, v, w] >= cut_val:
                    # compute gamma at [u, v, w]
                    gamma[u, v, w] = gamma_calc(dose_evaluation[u, v, w],
                    y_evaluation[u], x_evaluation[v], z_evaluation[w],
                    dose_reference[yi_start:yi_end + 1, xi_start:xi_end + 1,
                        zi_start:zi_end + 1], y_reference[yi_start:yi_end + 1],
                        x_reference[xi_start:xi_end + 1],
                        z_reference[zi_start:zi_end + 1], dose_crit, dist_crit,
                        search_dist, sampling, spacing, serchBox)

    return gamma