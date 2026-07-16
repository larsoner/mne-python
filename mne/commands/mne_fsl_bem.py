# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Create BEM surfaces using FSL (bet and betsurf).

Examples
--------
.. code-block:: console

    $ mne fsl_bem -s sample

"""

import sys

import mne
from mne.bem import make_fsl_bem


def run():
    """Run command."""
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__)

    parser.add_option(
        "-s", "--subject", dest="subject", help="Subject name (required)", default=None
    )
    parser.add_option(
        "-d",
        "--subjects-dir",
        dest="subjects_dir",
        help="Subjects directory",
        default=None,
    )
    parser.add_option(
        "-o",
        "--overwrite",
        dest="overwrite",
        help="Write over existing files",
        action="store_true",
    )
    parser.add_option(
        "-v", "--volume", dest="volume", help="Defaults to T1", default="T1"
    )
    parser.add_option(
        "-f",
        "--fraction",
        dest="fraction",
        default=None,
        type="float",
        help="Fractional intensity threshold, smaller values give larger "
        "brain outline estimates.",
    )
    parser.add_option(
        "-b",
        "--brainmask",
        dest="brainmask",
        help="Brainmask image to use instead of bet, e.g. brainmask.mgz "
        "(requires scikit-image).",
        default=None,
    )
    parser.add_option(
        "--smooth",
        dest="smooth",
        default=5,
        type="float",
        help="Amount to smooth the brainmask (in voxels). Only used when "
        "--brainmask is passed.",
    )
    parser.add_option(
        "--flirt",
        dest="flirt",
        help="Register to MNI152 using FSL's FLIRT instead of the "
        "FreeSurfer talairach transform.",
        action="store_true",
    )
    _add_verbose_flag(parser)

    options, args = parser.parse_args()

    if options.subject is None:
        parser.print_help()
        sys.exit(1)

    make_fsl_bem(
        subject=options.subject,
        subjects_dir=options.subjects_dir,
        overwrite=options.overwrite,
        volume=options.volume,
        brainmask=options.brainmask,
        smooth=options.smooth,
        fraction=options.fraction,
        talairach=not options.flirt,
        verbose=options.verbose,
    )


mne.utils.run_command_if_main()
