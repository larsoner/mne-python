# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD-3-Clause

import re
import numpy as np
import datetime

from ..base import BaseRaw
from ..meas_info import create_info, _format_dig_points
from ..utils import _mult_cal_one
from ...annotations import Annotations
<<<<<<< HEAD
from ...utils import logger, verbose, fill_doc, warn, _check_fname
from ...utils.check import _require_version
from ..constants import FIFF, FNIRS_SNIRF_DATATYPELABELS
=======
from ...utils import (logger, verbose, fill_doc, warn, _check_fname,
                      _import_h5py)
from ..constants import FIFF
>>>>>>> upstream/main
from .._digitization import _make_dig_points
from ...transforms import _frame_to_str, apply_trans
from ..nirx.nirx import _convert_fnirs_to_head
from ..._freesurfer import get_mni_fiducials

AVAILABLE_DATA_TYPES = [1, 201, 301, 99999]


@fill_doc
def read_raw_snirf(fname, optode_frame="unknown", preload=False, verbose=None):
    """Reader for a continuous wave SNIRF data.

    .. note:: This reader supports the .snirf file type only,
              not the .jnirs version.
              Files with either 3D or 2D locations can be read.
              However, we strongly recommend using 3D positions.
              If 2D positions are used the behaviour of MNE functions
              can not be guaranteed.

    Parameters
    ----------
    fname : str
        Path to the SNIRF data file.
    optode_frame : str
        Coordinate frame used for the optode positions. The default is unknown,
        in which case the positions are not modified. If a known coordinate
        frame is provided (head, meg, mri), then the positions are transformed
        in to the Neuromag head coordinate frame (head).
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawSNIRF
        A Raw object containing fNIRS data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawSNIRF(fname, optode_frame, preload, verbose)


def _open(fname):
    return open(fname, 'r', encoding='latin-1')


@fill_doc
class RawSNIRF(BaseRaw):
    """Raw object from a continuous wave SNIRF file.

    Parameters
    ----------
    fname : str
        Path to the SNIRF data file.
    optode_frame : str
        Coordinate frame used for the optode positions. The default is unknown,
        in which case the positions are not modified. If a known coordinate
        frame is provided (head, meg, mri), then the positions are transformed
        in to the Neuromag head coordinate frame (head).
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, optode_frame="unknown",
                 preload=False, verbose=None):
        # Must be here due to circular import error
        from ...preprocessing.nirs import _validate_nirs_info
        h5py = _import_h5py()

        fname = _check_fname(fname, 'read', True, 'fname')
        logger.info('Loading %s' % fname)

        with h5py.File(fname, 'r') as dat:

            if 'data2' in dat['nirs']:
                warn("File contains multiple recordings. "
                     "MNE does not support this feature. "
                     "Only the first dataset will be processed.")

<<<<<<< HEAD
            snirf_data_type = np.array(
                dat.get('nirs/data1/measurementList1/dataType'))
            if snirf_data_type not in AVAILABLE_DATA_TYPES:
                raise RuntimeError(
                    "File does not contain the supported data types. \
                    MNE only supports reading the following data types {}, \
                    but received type code {}. Processing is only available \
                    for data type 1 (CW data).".format(AVAILABLE_DATA_TYPES, snirf_data_type))
=======
            manafacturer = _get_metadata_str(dat, "ManufacturerName")
            if (optode_frame == "unknown") & (manafacturer == "Gowerlabs"):
                optode_frame = "head"

            snirf_data_type = np.array(dat.get('nirs/data1/measurementList1'
                                               '/dataType')).item()
            if snirf_data_type not in [1, 99999]:
                # 1 = Continuous Wave
                # 99999 = Processed
                raise RuntimeError('MNE only supports reading continuous'
                                   ' wave amplitude and processed haemoglobin'
                                   ' SNIRF files. Expected type'
                                   ' code 1 or 99999 but received type '
                                   f'code {snirf_data_type}')
>>>>>>> upstream/main

            last_samps = dat.get('/nirs/data1/dataTimeSeries').shape[0] - 1

            sampling_rate = _extract_sampling_rate(dat)

            if sampling_rate == 0:
                warn("Unable to extract sample rate from SNIRF file.")

            # Extract wavelengths
            fnirs_wavelengths = np.array(dat.get('nirs/probe/wavelengths'))
            fnirs_wavelengths = [int(w) for w in fnirs_wavelengths]
            if len(fnirs_wavelengths) != 2:
                raise RuntimeError(f'The data contains '
                                   f'{len(fnirs_wavelengths)}'
                                   f' wavelengths: {fnirs_wavelengths}. '
                                   f'MNE only supports reading continuous'
                                   ' wave amplitude SNIRF files '
                                   'with two wavelengths.')

            # Get data type specific probe information
            if snirf_data_type == 201:
                fnirs_time_delays = np.array(
                    dat.get('nirs/probe/timeDelays')).tolist()
                fnirs_time_delay_widths = np.array(
                    dat.get('nirs/probe/timeDelayWidths')).tolist()
            elif snirf_data_type == 301:
                fnirs_moment_orders = np.array(
                    dat.get('nirs/probe/momentOrders'))
                fnirs_moment_orders = [int(m) for m in fnirs_moment_orders]

            # Extract channels
            def atoi(text):
                return int(text) if text.isdigit() else text

            def natural_keys(text):
                return [atoi(c) for c in re.split(r'(\d+)', text)]

            channels = np.array([name for name in dat['nirs']['data1'].keys()])
            channels_idx = np.array(['measurementList' in n for n in channels])
            channels = channels[channels_idx]
            channels = sorted(channels, key=natural_keys)

            # Source and detector labels are optional fields.
            # Use S1, S2, S3, etc if not specified.
            if 'sourceLabels_disabled' in dat['nirs/probe']:
                # This is disabled as
                # MNE-Python does not currently support custom source names.
                # Instead, sources must be integer values.
                sources = np.array(dat.get('nirs/probe/sourceLabels'))
                sources = [s.decode('UTF-8') for s in sources]
            else:
                sources = np.unique([_correct_shape(np.array(dat.get(
                    'nirs/data1/' + c + '/sourceIndex')))[0]
                    for c in channels])
                sources = [f"S{int(s)}" for s in sources]

            if 'detectorLabels_disabled' in dat['nirs/probe']:
                # This is disabled as
                # MNE-Python does not currently support custom detector names.
                # Instead, detector must be integer values.
                detectors = np.array(dat.get('nirs/probe/detectorLabels'))
                detectors = [d.decode('UTF-8') for d in detectors]
            else:
                detectors = np.unique([_correct_shape(np.array(dat.get(
                    'nirs/data1/' + c + '/detectorIndex')))[0]
                    for c in channels])
                detectors = [f"D{int(d)}" for d in detectors]

            # Extract source and detector locations
            # 3D positions are optional in SNIRF,
            # but highly recommended in MNE.
            if ('detectorPos3D' in dat['nirs/probe']) &\
                    ('sourcePos3D' in dat['nirs/probe']):
                # If 3D positions are available they are used even if 2D exists
                detPos3D = np.array(dat.get('nirs/probe/detectorPos3D'))
                srcPos3D = np.array(dat.get('nirs/probe/sourcePos3D'))
            elif ('detectorPos2D' in dat['nirs/probe']) &\
                    ('sourcePos2D' in dat['nirs/probe']):
                warn('The data only contains 2D location information for the '
                     'optode positions. '
                     'It is highly recommended that data is used '
                     'which contains 3D location information for the '
                     'optode positions. With only 2D locations it can not be '
                     'guaranteed that MNE functions will behave correctly '
                     'and produce accurate results. If it is not possible to '
                     'include 3D positions in your data, please consider '
                     'using the set_montage() function.')

                detPos2D = np.array(dat.get('nirs/probe/detectorPos2D'))
                srcPos2D = np.array(dat.get('nirs/probe/sourcePos2D'))
                # Set the third dimension to zero. See gh#9308
                detPos3D = np.append(detPos2D,
                                     np.zeros((detPos2D.shape[0], 1)), axis=1)
                srcPos3D = np.append(srcPos2D,
                                     np.zeros((srcPos2D.shape[0], 1)), axis=1)

            else:
                raise RuntimeError('No optode location information is '
                                   'provided. MNE requires at least 2D '
                                   'location information')

            assert len(sources) == srcPos3D.shape[0]
            assert len(detectors) == detPos3D.shape[0]

            # Helper function for when the numpy array has shape (), i.e. just one element.
            def _correct_shape(arr):
                if arr.shape == ():
                    arr = arr[np.newaxis]
                return arr

            chnames = []
            ch_types = []
            for chan in channels:
<<<<<<< HEAD
                src_idx = int(
                    _correct_shape(
                        np.array(
                            dat.get(
                                'nirs/data1/' + chan + '/sourceIndex')))
                    [0])
                det_idx = int(
                    _correct_shape(
                        np.array(
                            dat.get(
                                'nirs/data1/' + chan + '/detectorIndex')))[0])
                if snirf_data_type == 1:
                    wve_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/wavelengthIndex')))[0])
=======
                src_idx = int(_correct_shape(np.array(dat.get('nirs/data1/' +
                                             chan + '/sourceIndex')))[0])
                det_idx = int(_correct_shape(np.array(dat.get('nirs/data1/' +
                                             chan + '/detectorIndex')))[0])

                if snirf_data_type == 1:
                    wve_idx = int(_correct_shape(np.array(
                        dat.get('nirs/data1/' + chan +
                                '/wavelengthIndex')))[0])
>>>>>>> upstream/main
                    ch_name = sources[src_idx - 1] + '_' +\
                        detectors[det_idx - 1] + ' ' +\
                        str(fnirs_wavelengths[wve_idx - 1])
                    chnames.append(ch_name)
<<<<<<< HEAD
                elif snirf_data_type == 201:
                    wve_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/wavelengthIndex')))[0])
                    bin_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/dataTypeIndex')))[0])
                    ch_name = sources[src_idx - 1] + '_' +\
                        detectors[det_idx - 1] + ' ' +\
                        str(fnirs_wavelengths[wve_idx - 1]) + ' ' +\
                        'bin' + str(fnirs_time_delays[bin_idx - 1])
                    chnames.append(ch_name)
                elif snirf_data_type == 301:
                    wve_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/wavelengthIndex')))[0])
                    moment_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/dataTypeIndex')))[0])
                    ch_name = sources[src_idx - 1] + '_' +\
                        detectors[det_idx - 1] + ' ' +\
                        str(fnirs_wavelengths[wve_idx - 1]) + ' ' +\
                        'moment' + str(fnirs_moment_orders[moment_idx - 1])
                    chnames.append(ch_name)
                elif snirf_data_type == 99999:
                    hb_id = _correct_shape(
                        np.array(dat.get('nirs/data1/' + chan + '/dataTypeLabel')))[0].decode('UTF-8')
                    ch_name = sources[src_idx - 1] + '_' +\
                        detectors[det_idx - 1] + ' ' +\
                        hb_id
                    chnames.append(ch_name)

            # Create mne structure
            if snirf_data_type == 1:
                info = create_info(chnames,
                                   sampling_rate,
                                   ch_types='fnirs_cw_amplitude')
            elif snirf_data_type == 201:
                info = create_info(chnames,
                                   sampling_rate,
                                   ch_types='fnirs_td_gated_amplitude')
            elif snirf_data_type == 301:
                info = create_info(chnames,
                                   sampling_rate,
                                   ch_types='fnirs_td_moments_amplitude')
            elif snirf_data_type == 99999:
                info = create_info(chnames,
                                   sampling_rate,
                                   ch_types='fnirs_processed')

            subject_info = {}
            names = _correct_shape(
                np.array(dat.get('nirs/metaDataTags/SubjectID')))
            subject_info['first_name'] = names[0].decode('UTF-8')
=======
                    ch_types.append('fnirs_cw_amplitude')

                elif snirf_data_type == 99999:
                    dt_id = _correct_shape(
                        np.array(dat.get('nirs/data1/' + chan +
                                         '/dataTypeLabel')))[0].decode('UTF-8')

                    # Convert between SNIRF processed names and MNE type names
                    dt_id = dt_id.lower().replace("dod", "fnirs_od")

                    ch_name = sources[src_idx - 1] + '_' + \
                        detectors[det_idx - 1]

                    if dt_id == "fnirs_od":
                        wve_idx = int(_correct_shape(np.array(
                            dat.get('nirs/data1/' + chan +
                                    '/wavelengthIndex')))[0])
                        suffix = ' ' + str(fnirs_wavelengths[wve_idx - 1])
                    else:
                        suffix = ' ' + dt_id.lower()
                    ch_name = ch_name + suffix

                    chnames.append(ch_name)
                    ch_types.append(dt_id)

            # Create mne structure
            info = create_info(chnames,
                               sampling_rate,
                               ch_types=ch_types)

            subject_info = {}
            names = np.array(dat.get('nirs/metaDataTags/SubjectID'))
            subject_info['first_name'] = \
                _correct_shape(names)[0].decode('UTF-8')
>>>>>>> upstream/main
            # Read non standard (but allowed) custom metadata tags
            if 'lastName' in dat.get('nirs/metaDataTags/'):
                ln = dat.get('/nirs/metaDataTags/lastName')[0].decode('UTF-8')
                subject_info['last_name'] = ln
            if 'middleName' in dat.get('nirs/metaDataTags/'):
                m = dat.get('/nirs/metaDataTags/middleName')[0].decode('UTF-8')
                subject_info['middle_name'] = m
            if 'sex' in dat.get('nirs/metaDataTags/'):
                s = dat.get('/nirs/metaDataTags/sex')[0].decode('UTF-8')
                if s in {'M', 'Male', '1', 'm'}:
                    subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_MALE
                elif s in {'F', 'Female', '2', 'f'}:
                    subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_FEMALE
                elif s in {'0', 'u'}:
                    subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
            # End non standard name reading
            # Update info
            info.update(subject_info=subject_info)

<<<<<<< HEAD
            LengthUnit = _correct_shape(
                np.array(dat.get('/nirs/metaDataTags/LengthUnit')))
            LengthUnit = LengthUnit[0].decode('UTF-8')
            scal = 1
            if "cm" in LengthUnit:
                scal = 100
            elif "mm" in LengthUnit:
                scal = 1000
=======
            length_unit = _get_metadata_str(dat, "LengthUnit")
            length_scaling = _get_lengthunit_scaling(length_unit)
>>>>>>> upstream/main

            srcPos3D /= length_scaling
            detPos3D /= length_scaling

            if optode_frame in ["mri", "meg"]:
                # These are all in MNI or MEG coordinates, so let's transform
                # them to the Neuromag head coordinate frame
                srcPos3D, detPos3D, _, head_t = _convert_fnirs_to_head(
                    'fsaverage', optode_frame, 'head', srcPos3D, detPos3D, [])
            else:
                head_t = np.eye(4)

            if optode_frame in ["head", "mri", "meg"]:
                # Then the transformation to head was performed above
                coord_frame = FIFF.FIFFV_COORD_HEAD
            elif 'MNE_coordFrame' in dat.get('nirs/metaDataTags/'):
                coord_frame = int(dat.get('/nirs/metaDataTags/MNE_coordFrame')
                                  [0])
            else:
                coord_frame = FIFF.FIFFV_COORD_UNKNOWN

            for idx, chan in enumerate(channels):
<<<<<<< HEAD
                src_idx = int(
                    _correct_shape(
                        np.array(
                            dat.get(
                                'nirs/data1/' + chan + '/sourceIndex')))
                    [0])
                det_idx = int(
                    _correct_shape(
                        np.array(
                            dat.get(
                                'nirs/data1/' + chan + '/detectorIndex')))[0])
=======
                src_idx = int(_correct_shape(np.array(dat.get('nirs/data1/' +
                                             chan + '/sourceIndex')))[0])
                det_idx = int(_correct_shape(np.array(dat.get('nirs/data1/' +
                                             chan + '/detectorIndex')))[0])

>>>>>>> upstream/main
                info['chs'][idx]['loc'][3:6] = srcPos3D[src_idx - 1, :]
                info['chs'][idx]['loc'][6:9] = detPos3D[det_idx - 1, :]
                # Store channel as mid point
                midpoint = (info['chs'][idx]['loc'][3:6] +
                            info['chs'][idx]['loc'][6:9]) / 2
                info['chs'][idx]['loc'][0:3] = midpoint
                info['chs'][idx]['coord_frame'] = coord_frame

<<<<<<< HEAD
                # get data type specific info:
                if snirf_data_type in [1, 201, 301]:
                    wve_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/wavelengthIndex')))[0])
                    info['chs'][idx]['loc'][9] = fnirs_wavelengths[wve_idx - 1]
                elif snirf_data_type == 99999:
                    hb_id = _correct_shape(
                        np.array(dat.get('nirs/data1/' + chan + '/dataTypeLabel')))[0].decode('UTF-8')
                    info['chs'][idx]['loc'][9] = FNIRS_SNIRF_DATATYPELABELS[hb_id]

                if snirf_data_type == 201:
                    bin_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/dataTypeIndex')))[0])
                    info['chs'][idx]['loc'][10] = fnirs_time_delays[bin_idx -
                                                                    1] * fnirs_time_delay_widths
                elif snirf_data_type == 301:
                    moment_idx = int(
                        _correct_shape(
                            np.array(
                                dat.get(
                                    'nirs/data1/' + chan +
                                    '/dataTypeIndex')))[0])
                    info['chs'][idx]['loc'][10] = fnirs_moment_orders[moment_idx - 1]
=======
                if (snirf_data_type in [1]) or \
                    ((snirf_data_type == 99999) and
                        (ch_types[idx] == "fnirs_od")):
                    wve_idx = int(_correct_shape(np.array(dat.get(
                        'nirs/data1/' + chan + '/wavelengthIndex')))[0])
                    info['chs'][idx]['loc'][9] = fnirs_wavelengths[wve_idx - 1]
>>>>>>> upstream/main

            if 'landmarkPos3D' in dat.get('nirs/probe/'):
                diglocs = np.array(dat.get('/nirs/probe/landmarkPos3D'))
                diglocs /= length_scaling
                digname = np.array(dat.get('/nirs/probe/landmarkLabels'))
                nasion, lpa, rpa, hpi = None, None, None, None
                extra_ps = dict()
                for idx, dign in enumerate(digname):
                    dign = dign.lower()
                    if dign in [b'lpa', b'al']:
                        lpa = diglocs[idx, :3]
                    elif dign in [b'nasion']:
                        nasion = diglocs[idx, :3]
                    elif dign in [b'rpa', b'ar']:
                        rpa = diglocs[idx, :3]
                    else:
                        extra_ps[f'EEG{len(extra_ps) + 1:03d}'] = \
                            diglocs[idx, :3]
                dig = _make_dig_points(
                    nasion=nasion, lpa=lpa, rpa=rpa, hpi=hpi,
                    dig_ch_pos=extra_ps,
                    coord_frame=_frame_to_str[coord_frame])
            else:
                ch_locs = [info['chs'][idx]['loc'][0:3]
                           for idx in range(len(channels))]
                # Set up digitization
                dig = get_mni_fiducials('fsaverage', verbose=False)
                for fid in dig:
                    fid['r'] = apply_trans(head_t, fid['r'])
                    fid['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                for ii, ch_loc in enumerate(ch_locs, 1):
                    dig.append(dict(
                        kind=FIFF.FIFFV_POINT_EEG,  # misnomer prob okay
                        r=ch_loc,
                        ident=ii,
                        coord_frame=FIFF.FIFFV_COORD_HEAD,
                    ))
                dig = _format_dig_points(dig)
                del head_t
            with info._unlock():
                info['dig'] = dig

<<<<<<< HEAD
                str_date = _correct_shape(np.array((dat.get(
                    '/nirs/metaDataTags/MeasurementDate'))))[0].decode('UTF-8')
                str_time = _correct_shape(np.array((dat.get(
                    '/nirs/metaDataTags/MeasurementTime'))))[0].decode('UTF-8')
                str_datetime = str_date + str_time
=======
            str_date = _correct_shape(np.array((dat.get(
                '/nirs/metaDataTags/MeasurementDate'))))[0].decode('UTF-8')
            str_time = _correct_shape(np.array((dat.get(
                '/nirs/metaDataTags/MeasurementTime'))))[0].decode('UTF-8')
            str_datetime = str_date + str_time
>>>>>>> upstream/main

                # Several formats have been observed so we try each in turn
                for dt_code in ['%Y-%m-%d%H:%M:%SZ',
                                '%Y-%m-%d%H:%M:%S']:
                    try:
                        meas_date = datetime.datetime.strptime(
                            str_datetime, dt_code)
                    except ValueError:
                        pass
                    else:
                        break
                else:
<<<<<<< HEAD
                    warn(
                        "Extraction of measurement date from SNIRF file failed. "
                        "The date is being set to January 1st, 2000, "
                        f"instead of {str_datetime}")
                    meas_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
                meas_date = meas_date.replace(tzinfo=datetime.timezone.utc)
                info['meas_date'] = meas_date

                if 'DateOfBirth' in dat.get('nirs/metaDataTags/'):
                    str_birth = _correct_shape(
                        np.array((dat.get('/nirs/metaDataTags/' 'DateOfBirth'))))[0].decode()
                    birth_matched = re.fullmatch(
                        r'(\d+)-(\d+)-(\d+)', str_birth)
                    if birth_matched is not None:
                        info["subject_info"]['birthday'] = (
                            int(birth_matched.groups()[0]),
                            int(birth_matched.groups()[1]),
                            int(birth_matched.groups()[2]))
=======
                    break
            else:
                warn("Extraction of measurement date from SNIRF file failed. "
                     "The date is being set to January 1st, 2000, "
                     f"instead of {str_datetime}")
                meas_date = datetime.datetime(2000, 1, 1, 0, 0, 0)
            meas_date = meas_date.replace(tzinfo=datetime.timezone.utc)
            with info._unlock():
                info['meas_date'] = meas_date

            if 'DateOfBirth' in dat.get('nirs/metaDataTags/'):
                str_birth = np.array((dat.get('/nirs/metaDataTags/'
                                              'DateOfBirth')))[0].decode()
                birth_matched = re.fullmatch(r'(\d+)-(\d+)-(\d+)', str_birth)
                if birth_matched is not None:
                    birthday = (int(birth_matched.groups()[0]),
                                int(birth_matched.groups()[1]),
                                int(birth_matched.groups()[2]))
                    with info._unlock():
                        info["subject_info"]['birthday'] = birthday
>>>>>>> upstream/main

            super(RawSNIRF, self).__init__(info, preload, filenames=[fname],
                                           last_samps=[last_samps],
                                           verbose=verbose)

            # Extract annotations
            annot = Annotations([], [], [])
            for key in dat['nirs']:
                if 'stim' in key:
                    data = np.atleast_2d(np.array(
                        dat.get('/nirs/' + key + '/data')))
                    if data.size > 0:
<<<<<<< HEAD
                        desc = _correct_shape(
                            np.array(dat.get('/nirs/' + key + '/name')))[0]
                        annot.append(data[:, 0], 1.0, desc.decode('UTF-8'))
            self.set_annotations(annot)

            # Reorder channels to match expected ordering in MNE
            if snirf_data_type in [1, 99999]:
                num_chans = len(self.ch_names)
                chans = []
                for idx in range(num_chans // 2):
                    chans.append(idx)
                    chans.append(idx + num_chans // 2)
                self.pick(picks=chans)
=======
                        desc = _correct_shape(np.array(dat.get(
                            '/nirs/' + key + '/name')))[0]
                        annot.append(data[:, 0], 1.0, desc.decode('UTF-8'))
            self.set_annotations(annot, emit_warning=False)
>>>>>>> upstream/main

        # Validate that the fNIRS info is correctly formatted
        _validate_nirs_info(self.info)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file."""
        import h5py

        with h5py.File(self._filenames[0], 'r') as dat:
            one = dat['/nirs/data1/dataTimeSeries'][start:stop].T

        _mult_cal_one(data, one, idx, cals, mult)


# Helper function for when the numpy array has shape (), i.e. just one element.
def _correct_shape(arr):
    if arr.shape == ():
        arr = arr[np.newaxis]
    return arr


def _get_timeunit_scaling(time_unit):
    """MNE expects time in seconds, return required scaling."""
    scalings = {'ms': 1000, 's': 1, 'unknown': 1}
    if time_unit in scalings:
        return scalings[time_unit]
    else:
        raise RuntimeError(f'The time unit {time_unit} is not supported by '
                           'MNE. Please report this error as a GitHub '
                           'issue to inform the developers.')


def _get_lengthunit_scaling(length_unit):
    """MNE expects distance in m, return required scaling."""
    scalings = {'m': 1, 'cm': 100, 'mm': 1000}
    if length_unit in scalings:
        return scalings[length_unit]
    else:
        raise RuntimeError(f'The length unit {length_unit} is not supported '
                           'by MNE. Please report this error as a GitHub '
                           'issue to inform the developers.')


def _extract_sampling_rate(dat):
    """Extract the sample rate from the time field."""
    time_data = np.array(dat.get('nirs/data1/time'))
    sampling_rate = 0
    if len(time_data) == 2:
        # specified as onset, samplerate
        sampling_rate = 1. / (time_data[1] - time_data[0])
    else:
        # specified as time points
        fs_diff = np.around(np.diff(time_data), decimals=4)
        if len(np.unique(fs_diff)) == 1:
            # Uniformly sampled data
            sampling_rate = 1. / np.unique(fs_diff)
        else:
            warn("MNE does not currently support reading "
                 "SNIRF files with non-uniform sampled data.")

    time_unit = _get_metadata_str(dat, "TimeUnit")
    time_unit_scaling = _get_timeunit_scaling(time_unit)
    sampling_rate *= time_unit_scaling

    return sampling_rate


def _get_metadata_str(dat, field):
    if field not in np.array(dat.get('nirs/metaDataTags')):
        return None
    data = dat.get(f'/nirs/metaDataTags/{field}')
    data = _correct_shape(np.array(data))
    data = str(data[0], 'utf-8')
    return data
