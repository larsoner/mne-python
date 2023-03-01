"""Generic wrapper function read_raw for specific read_raw_xxx readers."""

# Authors: Clemens Brunner <clemens.brunner@gmail.com>
#
# License: BSD-3-Clause


from pathlib import Path
from functools import partial

from . import (read_raw_edf, read_raw_bdf, read_raw_gdf, read_raw_brainvision,
               read_raw_fif, read_raw_eeglab, read_raw_cnt, read_raw_egi,
               read_raw_eximia, read_raw_nirx, read_raw_fieldtrip,
               read_raw_artemis123, read_raw_nicolet, read_raw_kit,
               read_raw_ctf, read_raw_boxy, read_raw_snirf, read_raw_fil)
from ..utils import fill_doc


def _read_unsupported(fname, **kwargs):
    ext = "".join(Path(fname).suffixes)
    msg = f"Unsupported file type ({ext})."
    suggest = kwargs.get("suggest")
    if suggest is not None:
        msg += f" Try reading a {suggest} file instead."
    msg += " Consider using a dedicated reader function for more options."
    raise ValueError(msg)


# supported read file formats
supported = {
    ".edf": dict(EDF=read_raw_edf),
    ".bdf": dict(BDF=read_raw_bdf),
    ".gdf": dict(GDF=read_raw_gdf),
    ".vhdr": dict(brainvision=read_raw_brainvision),
    ".ahdr": dict(brainvision=read_raw_brainvision),
    ".fif": dict(FIF=read_raw_fif),
    ".fif.gz": dict(FIF=read_raw_fif),
    ".set": dict(EEGLAB=read_raw_eeglab),
    ".cnt": dict(CNT=read_raw_cnt),
    ".mff": dict(EGI=read_raw_egi),
    ".nxe": dict(eximia=read_raw_eximia),
    ".hdr": dict(NIRx=read_raw_nirx),
    ".snirf": dict(SNIRF=read_raw_snirf),
    ".mat": dict(fieldtrip=read_raw_fieldtrip),
    ".bin": {
        'ARTEMIS': read_raw_artemis123,
        'UCL FIL OPM': read_raw_fil,
    },
    ".data": dict(Nicolet=read_raw_nicolet),
    ".sqd": dict(KIT=read_raw_kit),
    ".con": dict(KIT=read_raw_kit),
    ".ds": dict(CTF=read_raw_ctf),
    ".txt": dict(BOXY=read_raw_boxy),
}

# known but unsupported file formats
suggested = {
    ".vmrk": dict(brainvision=partial(_read_unsupported, suggest=".vhdr")),
    ".amrk": dict(brainvision=partial(_read_unsupported, suggest=".ahdr")),
    ".eeg": dict(brainvision=partial(_read_unsupported, suggest=".vhdr")),
}

# all known file formats
readers = {**supported, **suggested}


@fill_doc
def read_raw(fname, *, preload=False, verbose=None, **kwargs):
    """Read raw file.

    This function is a convenient wrapper for readers defined in `mne.io`. The
    correct reader is automatically selected based on the detected file format.
    All function arguments are passed to the respective reader.

    The following readers are currently supported:

    `~mne.io.read_raw_artemis123`, `~mne.io.read_raw_bdf`,
    `~mne.io.read_raw_boxy`, `~mne.io.read_raw_brainvision`,
    `~mne.io.read_raw_cnt`, `~mne.io.read_raw_ctf`, `~mne.io.read_raw_edf`,
    `~mne.io.read_raw_eeglab`, `~mne.io.read_raw_egi`,
    `~mne.io.read_raw_eximia`, `~mne.io.read_raw_fieldtrip`,
    `~mne.io.read_raw_fif`,  `~mne.io.read_raw_gdf`, `~mne.io.read_raw_kit`,
    `~mne.io.read_raw_fil`,
    `~mne.io.read_raw_nicolet`, and `~mne.io.read_raw_nirx`.

    Parameters
    ----------
    fname : path-like
        Name of the file to read.
    %(preload)s
    %(verbose)s
    **kwargs
        Additional keyword arguments to pass to the underlying reader. For
        details, see the arguments of the reader for the respective file
        format.

    Returns
    -------
    raw : mne.io.Raw
        Raw object.
    """
    ext = "".join(Path(fname).suffixes)
    kwargs['verbose'] = verbose
    kwargs['preload'] = preload
    if ext not in readers:
        _read_unsupported(fname)
    these_readers = list(readers[ext].values())
    for reader in these_readers:
        try:
            return reader(fname, **kwargs)
        except Exception:
            if len(these_readers) == 1:
                raise
    else:
        choices = '\n'.join(
            f'mne.io.{func.__name__.ljust(20)} ({kind})'
            for kind, func in readers[ext].items())
        raise RuntimeError(
            'Could not read file using any of the possible readers for '
            f'extension {ext}. Consider trying to read the file directly with '
            f'one of:\n{choices}')
