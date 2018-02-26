# -*- coding: utf-8 -*-
"""Traits-based GUI for head-MRI coregistration.

Hierarchy
---------
This is the hierarchy of classes for control. Brackets like [1] denote
properties that are set to be equivalent.

::

  CoregFrame: GUI for head-MRI coregistration.
  |-- CoregModel (model): Traits object for estimating the head mri transform.
  |   |-- MRIHeadWithFiducialsModel (mri) [1]: Represent an MRI head shape (high and low res) with fiducials.
  |   |   |-- SurfaceSource (bem_high_res): High-res MRI head
  |   |   |-- SurfaceSource (bem_low_res): Low-res MRI head
  |   |   +-- MRISubjectSource (subject_source) [2]: Find subjects in SUBJECTS_DIR and select one.
  |   |-- FiducialsSource (fid): Expose points of a given fiducials fif file.
  |   +-- DigSource (hsp): Expose measurement information from a inst file.
  |-- MlabSceneModel (scene) [3]: mayavi.core.ui.mayavi_scene
  |-- HeadViewController (headview) [4]: Set head views for the given coordinate system.
  |   +-- MlabSceneModel (scene) [3*]: ``HeadViewController(scene=CoregFrame.scene)``
  |-- SubjectSelectorPanel (subject_panel): Subject selector panel
  |   +-- MRISubjectSource (model) [2*]: ``SubjectSelectorPanel(model=self.model.mri.subject_source)``
  |-- SurfaceObject (mri_obj) [5]: Represent a solid object in a mayavi scene.
  |-- FiducialsPanel (fid_panel): Set fiducials on an MRI surface.
  |   |-- MRIHeadWithFiducialsModel (model) [1*]: ``FiducialsPanel(model=CoregFrame.model.mri, headview=CoregFrame.headview)``
  |   |-- HeadViewController (headview) [4*]: ``FiducialsPanel(model=CoregFrame.model.mri, headview=CoregFrame.headview)``
  |   +-- SurfaceObject (hsp_obj) [5*]: ``CoregFrame.fid_panel.hsp_obj = CoregFrame.mri_obj``
  |-- CoregPanel (coreg_panel): Coregistration panel for Head<->MRI with scaling.
  +-- PointObject ({hsp, eeg, lpa, nasion, rpa, hsp_lpa, hsp_nasion, hsp_rpa} + _obj): Represent a group of individual points in a mayavi scene.

MRI points and transformed via scaling, then by mri_head_t to the Neuromag
head coordinate frame.

Digitized points (in head coordinate frame) are never transformed.
"""  # noqa: E501

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
from ..externals.six.moves import queue
import re
from threading import Thread
import traceback
import warnings

import numpy as np

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pyface.api import (error, confirm, OK, YES, NO, CANCEL, information,
                        FileDialog, GUI)
from traits.api import (Bool, Button, cached_property, DelegatesTo, Directory,
                        Enum, Float, HasTraits, HasPrivateTraits, Instance,
                        Int, on_trait_change, Property, Str, List)
from traitsui.api import (View, Item, Group, HGroup, VGroup, VGrid, EnumEditor,
                          Handler, Label, TextEditor, Spring, InstanceEditor)
from traitsui.menu import Action, UndoButton, CancelButton, NoButtons
from tvtk.pyface.scene_editor import SceneEditor

from ..bem import make_bem_solution, write_bem_solution
from ..coreg import bem_fname, trans_fname
from ..defaults import DEFAULTS
from ..surface import _compute_nearest
from ..transforms import (write_trans, read_trans, apply_trans, rotation,
                          rotation_angles, Transform, _ensure_trans)
from ..coreg import fit_matched_points, scale_mri, _find_fiducials_files
from ..viz._3d import _toggle_mlab_render
from ..utils import logger, set_config
from ._fiducials_gui import MRIHeadWithFiducialsModel, FiducialsPanel, _mm_fmt
from ._file_traits import trans_wildcard, DigSource, SubjectSelectorPanel
from ._viewer import HeadViewController, PointObject, SurfaceObject

defaults = DEFAULTS['coreg']

laggy_float_editor = TextEditor(auto_set=False, enter_set=True, evaluate=float,
                                format_func=_mm_fmt)


class CoregModel(HasPrivateTraits):
    """Traits object for estimating the head mri transform.

    Notes
    -----
    Transform from head to mri space is modelled with the following steps:

    * move the head shape to its nasion position
    * rotate the head shape with user defined rotation around its nasion
    * move the head shape by user defined translation
    * move the head shape origin to the mri nasion

    If MRI scaling is enabled,

    * the MRI is scaled relative to its origin center (prior to any
      transformation of the digitizer head)

    Don't sync transforms to anything to prevent them from being recomputed
    upon every parameter change.
    """

    # data sources
    mri = Instance(MRIHeadWithFiducialsModel, ())
    hsp = Instance(DigSource, ())

    # parameters
    guess_mri_subject = Bool(True)  # change MRI subject when dig file changes
    grow_hair = Float(label="Grow Hair [mm]", desc="Move the back of the MRI "
                      "head outwards to compensate for hair on the digitizer "
                      "head shape")
    n_scale_params = Enum(0, 1, 3, desc="Scale the MRI to better fit the "
                          "subject's head shape (a new MRI subject will be "
                          "created with a name specified upon saving)")
    scale_x = Float(1, label="R (X)")
    scale_y = Float(1, label="A (Y)")
    scale_z = Float(1, label="S (Z)")
    rot_x = Float(0, label="R (X)")
    rot_y = Float(0, label="A (Y)")
    rot_z = Float(0, label="S (Z)")
    trans_x = Float(0, label="R (X)")
    trans_y = Float(0, label="A (Y)")
    trans_z = Float(0, label="S (Z)")
    parameters = List()
    lpa_weight = Float(1., label='Relative weight for LPA')
    nasion_weight = Float(1., label='Relative weight for Nasion')
    rpa_weight = Float(1., label='Relative weight for RPA')

    # options during scaling
    scale_labels = Bool(True, desc="whether to scale *.label files")
    copy_annot = Bool(True, desc="whether to copy *.annot files for scaled "
                      "subject")
    prepare_bem_model = Bool(True, desc="whether to run mne_prepare_bem_model "
                             "after scaling the MRI")

    # secondary to parameters
    scale = Property(depends_on=['n_scale_params'])
    has_fid_data = Property(
        Bool,
        desc="Required fiducials data is present.",
        depends_on=['mri:nasion', 'hsp:nasion'])
    has_pts_data = Property(
        Bool,
        depends_on=['mri:points', 'hsp:points'])
    has_eeg_data = Property(
        Bool,
        depends_on=['mri:points', 'hsp:eeg_points'])

    # target transforms
    mri_head_t = Property(
        desc="Transformaiton of the scaled MRI to the head coordinate frame.",
        depends_on=['parameters[]'])
    mri_trans = Property(depends_on=['parameters[]'])  # combo

    # info
    subject_has_bem = DelegatesTo('mri')
    lock_fiducials = DelegatesTo('mri')
    can_prepare_bem_model = Property(
        Bool,
        depends_on=['n_scale_params', 'subject_has_bem'])
    can_save = Property(Bool, depends_on=['mri_head_t'])
    raw_subject = Property(
        desc="Subject guess based on the raw file name.",
        depends_on=['hsp:inst_fname'])

    # transformed geometry
    processed_high_res_mri_points = Property(
        depends_on=['mri:bem_high_res:points', 'grow_hair'])
    processed_low_res_mri_points = Property(
        depends_on=['mri:bem_low_res:points', 'grow_hair'])
    transformed_high_res_mri_points = Property(
        depends_on=['processed_high_res_mri_points', 'mri_trans'])
    transformed_low_res_mri_points = Property(
        depends_on=['processed_low_res_mri_points', 'mri_trans'])
    nearest_transformed_low_res_mri_idx = Property(
        depends_on=['transformed_low_res_mri_points', 'hsp:points'])
    transformed_mri_lpa = Property(
        depends_on=['mri:lpa', 'mri_trans'])
    transformed_mri_nasion = Property(
        depends_on=['mri:nasion', 'mri_trans'])
    transformed_mri_rpa = Property(
        depends_on=['mri:rpa', 'mri_trans'])

    # fit properties
    lpa_distance = Property(
        depends_on=['transformed_mri_lpa', 'hsp:lpa'])
    nasion_distance = Property(
        depends_on=['transformed_mri_nasion', 'hsp:nasion'])
    rpa_distance = Property(
        depends_on=['transformed_mri_rpa', 'hsp:rpa'])
    point_distance = Property(  # use low res points
        depends_on=['nearest_transformed_low_res_mri_idx',
                    'transformed_low_res_mri_points', 'hsp:points'])

    # fit property info strings
    fid_eval_str = Property(
        depends_on=['lpa_distance', 'nasion_distance', 'rpa_distance'])
    points_eval_str = Property(
        depends_on=['point_distance'])

    @cached_property
    def _get_can_prepare_bem_model(self):
        return self.subject_has_bem and self.n_scale_params > 0

    @cached_property
    def _get_can_save(self):
        return np.any(self.mri_head_t != np.eye(4))

    @cached_property
    def _get_has_pts_data(self):
        has = (np.any(self.mri.bem_low_res.points) and
               np.any(self.hsp.points))
        return has

    @cached_property
    def _get_has_eeg_data(self):
        has = (np.any(self.mri.bem_low_res.points) and
               np.any(self.hsp.eeg_points))
        return has

    @cached_property
    def _get_has_fid_data(self):
        has = (np.any(self.mri.nasion) and np.any(self.hsp.nasion))
        return has

    @cached_property
    def _get_scale(self):
        if self.n_scale_params == 0:
            return np.array(1)
        return np.array([self.scale_x, self.scale_y, self.scale_z])

    @cached_property
    def _get_mri_head_t(self):
        if not self.has_fid_data:
            return np.eye(4)
        # rotate and translate hsp
        trans = rotation(self.rot_x, self.rot_y, self.rot_z)
        trans[:3, 3] = [self.trans_x, self.trans_y, self.trans_z]
        return trans

    @cached_property
    def _get_processed_high_res_mri_points(self):
        if self.grow_hair:
            if len(self.mri.bem_high_res.norms):
                scaled_hair_dist = self.grow_hair / (self.scale * 1000)
                points = self.mri.bem_high_res.points.copy()
                hair = points[:, 2] > points[:, 1]
                points[hair] += (self.mri.bem_high_res.norms[hair] *
                                 scaled_hair_dist)
                return points
            else:
                error(None, "Norms missing from bem, can't grow hair")
                self.grow_hair = 0
        else:
            return self.mri.bem_high_res.points

    @cached_property
    def _get_processed_low_res_mri_points(self):
        if self.grow_hair:
            if len(self.mri.bem_low_res.norms):
                scaled_hair_dist = self.grow_hair / (self.scale * 1000)
                points = self.mri.bem_low_res.points.copy()
                hair = points[:, 2] > points[:, 1]
                points[hair] += (self.mri.bem_low_res.norms[hair] *
                                 scaled_hair_dist)
                return points
            else:
                error(None, "Norms missing from bem, can't grow hair")
                self.grow_hair = 0
        else:
            return self.mri.bem_low_res.points

    @cached_property
    def _get_mri_trans(self):
        mri_scaling = np.ones(4)
        mri_scaling[:3] = self.scale
        return self.mri_head_t * mri_scaling

    @cached_property
    def _get_nearest_transformed_low_res_mri_idx(self):
        return _compute_nearest(self.transformed_low_res_mri_points,
                                self.hsp.points)

    @cached_property
    def _get_transformed_low_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_low_res_mri_points)
        return points

    @cached_property
    def _get_transformed_high_res_mri_points(self):
        points = apply_trans(self.mri_trans,
                             self.processed_high_res_mri_points)
        return points

    @cached_property
    def _get_transformed_mri_lpa(self):
        return apply_trans(self.mri_trans, self.mri.lpa)

    @cached_property
    def _get_transformed_mri_nasion(self):
        return apply_trans(self.mri_trans, self.mri.nasion)

    @cached_property
    def _get_transformed_mri_rpa(self):
        return apply_trans(self.mri_trans, self.mri.rpa)

    @cached_property
    def _get_lpa_distance(self):
        d = np.ravel(self.transformed_mri_lpa - self.hsp.lpa)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_nasion_distance(self):
        d = np.ravel(self.transformed_mri_nasion - self.hsp.nasion)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_rpa_distance(self):
        d = np.ravel(self.transformed_mri_rpa - self.hsp.rpa)
        return np.sqrt(np.dot(d, d))

    @cached_property
    def _get_point_distance(self):
        if (len(self.hsp.points) == 0 or
                len(self.transformed_low_res_mri_points) == 0):
            return None
        mri_points = self.transformed_low_res_mri_points[
            self.nearest_transformed_low_res_mri_idx]
        return np.linalg.norm(mri_points - self.hsp.points, axis=-1)

    @cached_property
    def _get_fid_eval_str(self):
        d = (self.lpa_distance * 1000, self.nasion_distance * 1000,
             self.rpa_distance * 1000)
        return 'Error: LPA=%.1f NAS=%.1f RPA=%.1f mm' % d

    @cached_property
    def _get_points_eval_str(self):
        if self.point_distance is None:
            return ""
        av_dist = 1000 * np.mean(self.point_distance)
        std_dist = 1000 * np.std(self.point_distance)
        return u"Points: μ=%.1f, σ=%.1f mm" % (av_dist, std_dist)

    def _get_raw_subject(self):
        # subject name guessed based on the inst file name
        if '_' in self.hsp.inst_fname:
            subject, _ = self.hsp.inst_fname.split('_', 1)
            if subject:
                return subject

    @on_trait_change('raw_subject')
    def _on_raw_subject_change(self, subject):
        if self.guess_mri_subject:
            if subject in self.mri.subject_source.subjects:
                self.mri.subject = subject
            elif 'fsaverage' in self.mri.subject_source.subjects:
                self.mri.subject = 'fsaverage'

    def omit_hsp_points(self, distance=0, reset=False):
        """Exclude head shape points that are far away from the MRI head.

        Parameters
        ----------
        distance : float
            Exclude all points that are further away from the MRI head than
            this distance. Previously excluded points are still excluded unless
            reset=True is specified. A value of distance <= 0 excludes nothing.
        reset : bool
            Reset the filter before calculating new omission (default is
            False).
        """
        distance = float(distance)
        if reset:
            logger.info("Coregistration: Reset excluded head shape points")
            with warnings.catch_warnings(record=True):  # Traits None comp
                self.hsp.points_filter = None

        if distance <= 0:
            return

        # find the new filter
        new_sub_filter = self.point_distance <= distance
        n_excluded = np.sum(new_sub_filter == False)  # noqa: E712
        logger.info("Coregistration: Excluding %i head shape points with "
                    "distance >= %.3f m.", n_excluded, distance)

        # combine the new filter with the previous filter
        old_filter = self.hsp.points_filter
        if old_filter is None:
            new_filter = new_sub_filter
        else:
            new_filter = np.ones(len(self.hsp.points), np.bool8)
            new_filter[old_filter] = new_sub_filter

        # set the filter
        with warnings.catch_warnings(record=True):  # comp to None in Traits
            self.hsp.points_filter = new_filter

    def fit_fiducials(self):
        """Find rotation and translation to fit all 3 fiducials."""
        head_pts = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        mri_pts = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        mri_pts *= self.scale
        weights = [self.lpa_weight, self.nasion_weight, self.rpa_weight]

        x0 = (self.rot_x, self.rot_y, self.rot_z, self.trans_x, self.trans_y,
              self.trans_z)
        est = fit_matched_points(mri_pts, head_pts, x0=x0, out='params',
                                 weights=weights)
        self.parameters[:6] = est

    def fit_icp(self):
        """Find rotation and translation to fit head shapes (ICP)."""
        head_pts = np.concatenate([
            self.hsp.points,
            self.hsp.lpa, self.hsp.nasion, self.hsp.rpa])
        mri_pts = np.concatenate([
            self.processed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx],
            self.mri.lpa, self.mri.nasion, self.mri.rpa])
        mri_pts *= self.scale
        weights = np.ones(mri_pts.shape[0])
        weights[-3:] = [self.lpa_weight, self.nasion_weight, self.rpa_weight]

        x0 = (self.rot_x, self.rot_y, self.rot_z,
              self.trans_x, self.trans_y, self.trans_z)
        est = fit_matched_points(mri_pts, head_pts, x0=x0, out='params',
                                 weights=weights)

        self.parameters[:6] = est

    def fit_scale_fiducials(self):
        """Find translation, rotation, scaling based on the three fiducials."""
        head_fid = np.vstack((self.hsp.lpa, self.hsp.nasion, self.hsp.rpa))
        mri_fid = np.vstack((self.mri.lpa, self.mri.nasion, self.mri.rpa))
        weights = [self.lpa_weight, self.nasion_weight, self.rpa_weight]

        x0 = (self.rot_x, self.rot_y, self.rot_z, self.trans_x, self.trans_y,
              self.trans_z, self.scale_x,)
        est = fit_matched_points(mri_fid, head_fid, scale=1, x0=x0,
                                 out='params', weights=weights)

        self.parameters[:] = np.concatenate([est, [est[-1]] * 2])

    def fit_scale_icp(self):
        """Find MRI scaling, translation, and rotation to match HSP."""
        head_pts = np.concatenate([
            self.hsp.points,
            self.hsp.lpa, self.hsp.nasion, self.hsp.rpa])
        mri_pts = np.concatenate([
            self.processed_low_res_mri_points[
                self.nearest_transformed_low_res_mri_idx],
            self.mri.lpa, self.mri.nasion, self.mri.rpa])
        weights = np.ones(mri_pts.shape[0])
        weights[-3:] = [self.lpa_weight, self.nasion_weight, self.rpa_weight]
        if self.n_scale_params == 1:
            x0 = (self.rot_x, self.rot_y, self.rot_z,
                  self.trans_x, self.trans_y, self.trans_z,
                  self.scale_x)
            est = fit_matched_points(mri_pts, head_pts, scale=1, x0=x0,
                                     out='params', weights=weights)
            est = np.concatenate([est, [est[-1]] * 2])
        else:  # if self.n_scale_params == 3:
            x0 = (self.rot_x, self.rot_y, self.rot_z,
                  self.trans_x, self.trans_y, self.trans_z,
                  1. / self.scale_x, 1. / self.scale_y, 1. / self.scale_z)
            est = fit_matched_points(mri_pts, head_pts, scale=3, x0=x0,
                                     out='params', weights=weights)
        self.parameters[:] = est

    def get_scaling_job(self, subject_to, skip_fiducials):
        """Find all arguments needed for the scaling worker."""
        subjects_dir = self.mri.subjects_dir
        subject_from = self.mri.subject
        bem_names = []
        if self.can_prepare_bem_model and self.prepare_bem_model:
            pattern = bem_fname.format(subjects_dir=subjects_dir,
                                       subject=subject_from, name='(.+-bem)')
            bem_dir, pattern = os.path.split(pattern)
            for filename in os.listdir(bem_dir):
                match = re.match(pattern, filename)
                if match:
                    bem_names.append(match.group(1))

        return (subjects_dir, subject_from, subject_to, self.scale,
                skip_fiducials, self.scale_labels, self.copy_annot, bem_names)

    def load_trans(self, fname):
        """Load the head-mri transform from a fif file.

        Parameters
        ----------
        fname : str
            File path.
        """
        self.set_trans(_ensure_trans(read_trans(fname, return_all=True),
                                     'mri', 'head')['trans'])

    def reset(self):
        """Reset all the parameters affecting the coregistration."""
        self.reset_traits(('grow_hair', 'n_scaling_params', 'scale_x',
                           'scale_y', 'scale_z', 'rot_x', 'rot_y', 'rot_z',
                           'trans_x', 'trans_y', 'trans_z'))

    def set_trans(self, mri_head_t):
        """Set rotation and translation params from a transformation matrix.

        Parameters
        ----------
        mri_head_t : array, shape (4, 4)
            Transformation matrix from MRI to head space.
        """
        rot_x, rot_y, rot_z = rotation_angles(mri_head_t)
        x, y, z = mri_head_t[:3, 3]
        self.parameters[:6] = [rot_x, rot_y, rot_z, x, y, z]

    def save_trans(self, fname):
        """Save the head-mri transform as a fif file.

        Parameters
        ----------
        fname : str
            Target file path.
        """
        if not self.can_save:
            raise RuntimeError("Not enough information for saving transform")
        write_trans(fname, Transform('mri', 'head', self.mri_head_t))

    def _parameters_items_changed(self):
        # Update rot_x, rot_y, rot_z parameters if necessary
        for ii, key in enumerate(('rot_x', 'rot_y', 'rot_z',
                                  'trans_x', 'trans_y', 'trans_z',
                                  'scale_x', 'scale_y', 'scale_z')):
            if self.parameters[ii] != getattr(self, key):  # prevent circular
                setattr(self, key, self.parameters[ii])

    def _rot_x_changed(self):
        self.parameters[0] = self.rot_x

    def _rot_y_changed(self):
        self.parameters[1] = self.rot_y

    def _rot_z_changed(self):
        self.parameters[2] = self.rot_z

    def _trans_x_changed(self):
        self.parameters[3] = self.trans_x

    def _trans_y_changed(self):
        self.parameters[4] = self.trans_y

    def _trans_z_changed(self):
        self.parameters[5] = self.trans_z

    def _scale_x_changed(self):
        self.parameters[6] = self.scale_x

    def _scale_y_changed(self):
        self.parameters[7] = self.scale_y

    def _scale_z_changed(self):
        self.parameters[8] = self.scale_z


class CoregFrameHandler(Handler):
    """Check for unfinished processes before closing its window."""

    def object_title_changed(self, info):
        """Set the title when it gets changed."""
        info.ui.title = info.object.title

    def close(self, info, is_ok):
        """Handle the close event."""
        if info.object.queue.unfinished_tasks:
            information(None, "Can not close the window while saving is still "
                        "in progress. Please wait until all MRIs are "
                        "processed.", "Saving Still in Progress")
            return False
        else:
            # store configuration, but don't prevent from closing on error
            try:
                info.object.save_config()
            except Exception as exc:
                warnings.warn("Error saving GUI configuration:\n%s" % (exc,))
            return True


def _make_view_coreg_panel(scrollable=False):
    """Generate View for CoregPanel."""
    view = View(VGroup(Item('grow_hair', show_label=True),
                       Item('n_scale_params', label='MRI Scaling',
                            style='custom', show_label=True,
                            editor=EnumEditor(values={0: '1:None',
                                                      1: '2:Uniform',
                                                      3: '3:3-axis'},
                                              cols=4)),
                       VGrid(Item('scale_x', editor=laggy_float_editor,
                                  show_label=True, tooltip="Scale along "
                                  "right-left axis",
                                  enabled_when='n_scale_params > 0',
                                  width=+50),
                             Item('scale_x_dec',
                                  enabled_when='n_scale_params > 0',
                                  width=-50),
                             Item('scale_x_inc',
                                  enabled_when='n_scale_params > 0',
                                  width=-50),
                             Item('scale_step', tooltip="Scaling step",
                                  enabled_when='n_scale_params > 0',
                                  width=+50),
                             Item('scale_y', editor=laggy_float_editor,
                                  show_label=True,
                                  enabled_when='n_scale_params > 1',
                                  tooltip="Scale along anterior-posterior "
                                  "axis", width=+50),
                             Item('scale_y_dec',
                                  enabled_when='n_scale_params > 1',
                                  width=-50),
                             Item('scale_y_inc',
                                  enabled_when='n_scale_params > 1',
                                  width=-50),
                             Label('(Step)', width=+50),
                             Item('scale_z', editor=laggy_float_editor,
                                  show_label=True,
                                  enabled_when='n_scale_params > 1',
                                  tooltip="Scale along anterior-posterior "
                                  "axis", width=+50),
                             Item('scale_z_dec',
                                  enabled_when='n_scale_params > 1',
                                  width=-50),
                             Item('scale_z_inc',
                                  enabled_when='n_scale_params > 1',
                                  width=-50),
                             show_labels=False, show_border=True,
                             label='Scaling', columns=4),
                       HGroup(Item('fits_icp',
                                   enabled_when='n_scale_params',
                                   tooltip="Rotate, translate, and scale the "
                                   "MRI to minimize the distance from each "
                                   "digitizer point to the closest MRI point "
                                   "(one ICP iteration)"),
                              Item('fits_fid',
                                   enabled_when='n_scale_params == 1',
                                   tooltip="Rotate, translate, and scale the "
                                   "MRI to minimize the distance of the three "
                                   "fiducials."),
                              show_labels=False),
                       VGrid(Item('trans_x', editor=laggy_float_editor,
                                  show_label=True, tooltip="Move along "
                                  "right-left axis", width=+50),
                             Item('trans_x_dec', width=-50),
                             Item('trans_x_inc', width=-50),
                             Item('trans_step', tooltip="Movement step",
                                  width=+50),
                             Item('trans_y', editor=laggy_float_editor,
                                  show_label=True, tooltip="Move along "
                                  "anterior-posterior axis", width=+50),
                             Item('trans_y_dec', width=-50),
                             Item('trans_y_inc', width=-50),
                             Label('(Step)', width=+50),
                             Item('trans_z', editor=laggy_float_editor,
                                  show_label=True, tooltip="Move along "
                                  "anterior-posterior axis", width=+50),
                             Item('trans_z_dec', width=-50),
                             Item('trans_z_inc', width=-50),
                             show_labels=False, show_border=True,
                             label='Translation', columns=4),
                       VGrid(Item('rot_x', editor=laggy_float_editor,
                                  show_label=True, tooltip="Rotate along "
                                  "right-left axis", width=+50),
                             Item('rot_x_dec', width=-50),
                             Item('rot_x_inc', width=-50),
                             Item('rot_step', tooltip="Rotation step",
                                  width=+50),
                             Item('rot_y', editor=laggy_float_editor,
                                  show_label=True, tooltip="Rotate along "
                                  "anterior-posterior axis", width=+50),
                             Item('rot_y_dec', width=-50),
                             Item('rot_y_inc', width=-50),
                             Label('(Step)', width=+50),
                             Item('rot_z', editor=laggy_float_editor,
                                  show_label=True, tooltip="Rotate along "
                                  "anterior-posterior axis", width=+50),
                             Item('rot_z_dec', width=-50),
                             Item('rot_z_inc', width=-50),
                             show_labels=False, show_border=True,
                             label='Rotation', columns=4),
                       # buttons
                       HGroup(Item('fit_icp',
                                   enabled_when='has_pts_data',
                                   tooltip="Rotate and translate the "
                                   "MRI to minimize the distance from each "
                                   "digitizer point to the closest MRI point "
                                   "(one ICP iteration)", width=10),
                              Item('fit_fid', enabled_when='has_fid_data',
                                   tooltip="Rotate and translate the "
                                   "MRI to minimize the distance of the three "
                                   "fiducials.", width=10),
                              show_labels=False),
                       # Fitting weights
                       VGrid(Item('lpa_weight', editor=laggy_float_editor,
                                  show_label=True, tooltip="Relative weight "
                                  "for LPA", label='LPA'),
                             Item('nasion_weight', editor=laggy_float_editor,
                                  show_label=True, tooltip="Relative weight "
                                  "for nasion", label='Na.'),
                             Item('rpa_weight', editor=laggy_float_editor,
                                  show_label=True, tooltip="Relative weight "
                                  "for RPA", label='RPA'),
                             show_labels=False, show_border=True,
                             label='Fitting weights',
                             columns=3),
                       # Trans
                       HGroup(Item('load_trans', width=10),
                              Spring(), show_labels=False),
                       '_',
                       Item('fid_eval_str', style='readonly'),
                       Item('points_eval_str', style='readonly'),
                       '_',
                       VGroup(
                           Item('scale_labels',
                                label="Scale *.label files",
                                enabled_when='n_scale_params > 0'),
                           Item('copy_annot',
                                label="Copy annotation files",
                                enabled_when='n_scale_params > 0'),
                           Item('prepare_bem_model',
                                label="Run mne_prepare_bem_model",
                                enabled_when='can_prepare_bem_model'),
                           show_left=False,
                           label='Scaling options',
                           show_border=True),
                       '_',
                       HGroup(Item('save', enabled_when='can_save',
                                   tooltip="Save the trans file and (if "
                                   "scaling is enabled) the scaled MRI"),
                              Item('reset_params', tooltip="Reset all "
                                   "coregistration parameters"),
                              show_labels=False),
                       Item('queue_feedback', style='readonly'),
                       Item('queue_current', style='readonly'),
                       Item('queue_len_str', style='readonly'),
                       show_labels=False),
                kind='panel', buttons=[UndoButton], scrollable=scrollable)
    return view


class CoregPanel(HasPrivateTraits):
    """Coregistration panel for Head<->MRI with scaling."""

    model = Instance(CoregModel)

    # parameters
    reset_params = Button(label='Reset')
    grow_hair = DelegatesTo('model')
    n_scale_params = DelegatesTo('model')
    scale_step = Float(0.01)
    scale_x = DelegatesTo('model')
    scale_x_dec = Button('-')
    scale_x_inc = Button('+')
    scale_y = DelegatesTo('model')
    scale_y_dec = Button('-')
    scale_y_inc = Button('+')
    scale_z = DelegatesTo('model')
    scale_z_dec = Button('-')
    scale_z_inc = Button('+')
    rot_step = Float(0.01)
    rot_x = DelegatesTo('model')
    rot_x_dec = Button('-')
    rot_x_inc = Button('+')
    rot_y = DelegatesTo('model')
    rot_y_dec = Button('-')
    rot_y_inc = Button('+')
    rot_z = DelegatesTo('model')
    rot_z_dec = Button('-')
    rot_z_inc = Button('+')
    trans_step = Float(0.001)
    trans_x = DelegatesTo('model')
    trans_x_dec = Button('-')
    trans_x_inc = Button('+')
    trans_y = DelegatesTo('model')
    trans_y_dec = Button('-')
    trans_y_inc = Button('+')
    trans_z = DelegatesTo('model')
    trans_z_dec = Button('-')
    trans_z_inc = Button('+')
    lpa_weight = DelegatesTo('model')
    nasion_weight = DelegatesTo('model')
    rpa_weight = DelegatesTo('model')

    # fitting
    has_fid_data = DelegatesTo('model')
    has_pts_data = DelegatesTo('model')
    has_eeg_data = DelegatesTo('model')
    # fitting with scaling
    fits_icp = Button(label='Fit (ICP)')
    fits_fid = Button(label='Fit Fiducials')
    # fitting without scaling
    fit_icp = Button(label='Fit (ICP)')
    fit_fid = Button(label='Fit Fiducials')

    # fit info
    fid_eval_str = DelegatesTo('model')
    points_eval_str = DelegatesTo('model')

    # saving
    can_prepare_bem_model = DelegatesTo('model')
    can_save = DelegatesTo('model')
    scale_labels = DelegatesTo('model')
    copy_annot = DelegatesTo('model')
    prepare_bem_model = DelegatesTo('model')
    save = Button(label="Save Subject As...")
    load_trans = Button(label='Load trans...')
    queue = Instance(queue.Queue, ())
    queue_feedback = Str('')
    queue_current = Str('')
    queue_len = Int(0)
    queue_len_str = Property(Str, depends_on=['queue_len'])

    view = _make_view_coreg_panel()

    def __init__(self, *args, **kwargs):  # noqa: D102
        super(CoregPanel, self).__init__(*args, **kwargs)
        self.model.parameters = [0., 0., 0., 0., 0., 0., 1., 1., 1.]

        # Setup scaling worker
        def worker():
            while True:
                (subjects_dir, subject_from, subject_to, scale, skip_fiducials,
                 include_labels, include_annot, bem_names) = self.queue.get()
                self.queue_len -= 1

                # Scale MRI files
                self.queue_current = 'Scaling %s...' % subject_to
                try:
                    scale_mri(subject_from, subject_to, scale, True,
                              subjects_dir, skip_fiducials, include_labels,
                              include_annot)
                except Exception:
                    logger.error('Error scaling %s:\n' % subject_to +
                                 traceback.format_exc())
                    self.queue_feedback = ('Error scaling %s (see Terminal)' %
                                           subject_to)
                    bem_names = ()  # skip bem solutions
                else:
                    self.queue_feedback = 'Done scaling %s.' % subject_to

                # Precompute BEM solutions
                for bem_name in bem_names:
                    self.queue_current = ('Computing %s solution...' %
                                          bem_name)
                    try:
                        bem_file = bem_fname.format(subjects_dir=subjects_dir,
                                                    subject=subject_to,
                                                    name=bem_name)
                        bemsol = make_bem_solution(bem_file)
                        write_bem_solution(bem_file[:-4] + '-sol.fif', bemsol)
                    except Exception:
                        logger.error('Error computing %s solution:\n' %
                                     bem_name + traceback.format_exc())
                        self.queue_feedback = ('Error computing %s solution '
                                               '(see Terminal)' % bem_name)
                    else:
                        self.queue_feedback = ('Done computing %s solution.' %
                                               bem_name)

                # Finalize
                self.queue_current = ''
                self.queue.task_done()

        t = Thread(target=worker)
        t.daemon = True
        t.start()

    @cached_property
    def _get_queue_len_str(self):
        if self.queue_len:
            return "Queue length: %i" % self.queue_len
        else:
            return ''

    @cached_property
    def _get_rotation(self):
        rot = np.array([self.rot_x, self.rot_y, self.rot_z])
        return rot

    @cached_property
    def _get_src_pts(self):
        return self.hsp_pts - self.hsp_fid[0]

    @cached_property
    def _get_src_fid(self):
        return self.hsp_fid - self.hsp_fid[0]

    @cached_property
    def _get_tgt_origin(self):
        return self.mri_fid[0] * self.scale

    @cached_property
    def _get_tgt_pts(self):
        pts = self.mri_pts * self.scale
        pts -= self.tgt_origin
        return pts

    @cached_property
    def _get_tgt_fid(self):
        fid = self.mri_fid * self.scale
        fid -= self.tgt_origin
        return fid

    @cached_property
    def _get_translation(self):
        trans = np.array([self.trans_x, self.trans_y, self.trans_z])
        return trans

    def _fit_fid_fired(self):
        GUI.set_busy()
        self.model.fit_fiducials()
        GUI.set_busy(False)

    def _fit_icp_fired(self):
        GUI.set_busy()
        self.model.fit_icp()
        GUI.set_busy(False)

    def _fits_fid_fired(self):
        GUI.set_busy()
        self.model.fit_scale_fiducials()
        GUI.set_busy(False)

    def _fits_icp_fired(self):
        GUI.set_busy()
        self.model.fit_scale_icp()
        GUI.set_busy(False)

    def _reset_params_fired(self):
        self.model.reset()

    def _rot_x_dec_fired(self):
        self.rot_x -= self.rot_step

    def _rot_x_inc_fired(self):
        self.rot_x += self.rot_step

    def _rot_y_dec_fired(self):
        self.rot_y -= self.rot_step

    def _rot_y_inc_fired(self):
        self.rot_y += self.rot_step

    def _rot_z_dec_fired(self):
        self.rot_z -= self.rot_step

    def _rot_z_inc_fired(self):
        self.rot_z += self.rot_step

    def _load_trans_fired(self):
        # find trans file destination
        raw_dir = os.path.dirname(self.model.hsp.file)
        subject = self.model.mri.subject
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject)
        dlg = FileDialog(action="open", wildcard=trans_wildcard,
                         default_path=trans_file)
        if dlg.open() != OK:
            return
        trans_file = dlg.path
        try:
            self.model.load_trans(trans_file)
        except Exception as e:
            error(None, "Error loading trans file %s: %s (See terminal "
                  "for details)" % (trans_file, e), "Error Loading Trans File")
            raise

    def _save_fired(self):
        subjects_dir = self.model.mri.subjects_dir
        subject_from = self.model.mri.subject

        # check that fiducials are saved
        skip_fiducials = False
        if self.n_scale_params and not _find_fiducials_files(subject_from,
                                                             subjects_dir):
            msg = ("No fiducials file has been found for {src}. If fiducials "
                   "are not saved, they will not be available in the scaled "
                   "MRI. Should the current fiducials be saved now? "
                   "Select Yes to save the fiducials at "
                   "{src}/bem/{src}-fiducials.fif. "
                   "Select No to proceed scaling the MRI without fiducials.".
                   format(src=subject_from))
            title = "Save Fiducials for %s?" % subject_from
            rc = confirm(None, msg, title, cancel=True, default=CANCEL)
            if rc == CANCEL:
                return
            elif rc == YES:
                self.model.mri.save(self.model.mri.default_fid_fname)
            elif rc == NO:
                skip_fiducials = True
            else:
                raise RuntimeError("rc=%s" % repr(rc))

        # find target subject
        if self.n_scale_params:
            subject_to = self.model.raw_subject or subject_from
            mridlg = NewMriDialog(subjects_dir=subjects_dir,
                                  subject_from=subject_from,
                                  subject_to=subject_to)
            ui = mridlg.edit_traits(kind='modal')
            if not ui.result:  # i.e., user pressed cancel
                return
            subject_to = mridlg.subject_to
        else:
            subject_to = subject_from

        # find trans file destination
        raw_dir = os.path.dirname(self.model.hsp.file)
        trans_file = trans_fname.format(raw_dir=raw_dir, subject=subject_to)
        dlg = FileDialog(action="save as", wildcard=trans_wildcard,
                         default_path=trans_file)
        dlg.open()
        if dlg.return_code != OK:
            return
        trans_file = dlg.path
        if not trans_file.endswith('.fif'):
            trans_file += '.fif'
            if os.path.exists(trans_file):
                answer = confirm(None, "The file %r already exists. Should it "
                                 "be replaced?", "Overwrite File?")
                if answer != YES:
                    return

        # save the trans file
        try:
            self.model.save_trans(trans_file)
        except Exception as e:
            error(None, "Error saving -trans.fif file: %s (See terminal for "
                  "details)" % (e,), "Error Saving Trans File")
            raise

        # save the scaled MRI
        if self.n_scale_params:
            job = self.model.get_scaling_job(subject_to, skip_fiducials)
            self.queue.put(job)
            self.queue_len += 1

    def _scale_x_dec_fired(self):
        self.scale_x -= self.scale_step

    def _scale_x_inc_fired(self):
        self.scale_x += self.scale_step

    def _scale_y_dec_fired(self):
        self.scale_y -= self.scale_step

    def _scale_y_inc_fired(self):
        self.scale_y += self.scale_step

    def _scale_z_dec_fired(self):
        self.scale_z -= self.scale_step

    def _scale_z_inc_fired(self):
        self.scale_z += self.scale_step

    def _trans_x_dec_fired(self):
        self.trans_x -= self.trans_step

    def _trans_x_inc_fired(self):
        self.trans_x += self.trans_step

    def _trans_y_dec_fired(self):
        self.trans_y -= self.trans_step

    def _trans_y_inc_fired(self):
        self.trans_y += self.trans_step

    def _trans_z_dec_fired(self):
        self.trans_z -= self.trans_step

    def _trans_z_inc_fired(self):
        self.trans_z += self.trans_step


class NewMriDialog(HasPrivateTraits):
    """New MRI dialog."""

    # Dialog to determine target subject name for a scaled MRI
    subjects_dir = Directory
    subject_to = Str
    subject_from = Str
    subject_to_dir = Property(depends_on=['subjects_dir', 'subject_to'])
    subject_to_exists = Property(Bool, depends_on='subject_to_dir')

    feedback = Str(' ' * 100)
    can_overwrite = Bool
    overwrite = Bool
    can_save = Bool

    view = View(Item('subject_to', label='New MRI Subject Name', tooltip="A "
                     "new folder with this name will be created in the "
                     "current subjects_dir for the scaled MRI files"),
                Item('feedback', show_label=False, style='readonly'),
                Item('overwrite', enabled_when='can_overwrite', tooltip="If a "
                     "subject with the chosen name exists, delete the old "
                     "subject"),
                buttons=[CancelButton,
                         Action(name='OK', enabled_when='can_save')])

    def _can_overwrite_changed(self, new):
        if not new:
            self.overwrite = False

    @cached_property
    def _get_subject_to_dir(self):
        return os.path.join(self.subjects_dir, self.subject_to)

    @cached_property
    def _get_subject_to_exists(self):
        if not self.subject_to:
            return False
        elif os.path.exists(self.subject_to_dir):
            return True
        else:
            return False

    @on_trait_change('subject_to_dir,overwrite')
    def update_dialog(self):
        if not self.subject_from:
            # weird trait state that occurs even when subject_from is set
            return
        elif not self.subject_to:
            self.feedback = "No subject specified..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject_to == self.subject_from:
            self.feedback = "Must be different from MRI source subject..."
            self.can_save = False
            self.can_overwrite = False
        elif self.subject_to_exists:
            if self.overwrite:
                self.feedback = "%s will be overwritten." % self.subject_to
                self.can_save = True
                self.can_overwrite = True
            else:
                self.feedback = "Subject already exists..."
                self.can_save = False
                self.can_overwrite = True
        else:
            self.feedback = "Name ok."
            self.can_save = True
            self.can_overwrite = False


def _make_view(tabbed=False, split=False, scene_width=500, scene_height=400,
               scrollable=True):
    """Create a view for the CoregFrame.

    Parameters
    ----------
    tabbed : bool
        Combine the data source panel and the coregistration panel into a
        single panel with tabs.
    split : bool
        Split the main panels with a movable splitter (good for QT4 but
        unnecessary for wx backend).
    scene_width : int
        Specify a minimum width for the 3d scene (in pixels).
    scrollable : bool
        Make the coregistration panel vertically scrollable (default True).

    Returns
    -------
    view : traits View
        View object for the CoregFrame.
    """
    scene = VGroup(
        Item('scene', show_label=False,
             editor=SceneEditor(scene_class=MayaviScene),
             dock='vertical', width=scene_width, height=scene_height),
        VGroup(
            Item('headview', style='custom'),
            'view_options',
            show_border=True, show_labels=False, label='View'))

    data_panel = VGroup(
        VGroup(Item('subject_panel', style='custom'), label="MRI Subject",
               show_border=True, show_labels=False),
        VGroup(Item('lock_fiducials', style='custom',
                    editor=EnumEditor(cols=2, values={False: '2:Edit',
                                                      True: '1:Lock'}),
                    enabled_when='fid_ok'),
               HGroup('hsp_always_visible',
                      Label("Always Show Head Shape Points"),
                      show_labels=False),
               Item('fid_panel', style='custom'),
               label="MRI Fiducials",  show_border=True, show_labels=False),
        VGroup(Item('raw_src', style="custom"),
               HGroup('guess_mri_subject',
                      Label('Guess MRI Subject from File Name'),
                      show_labels=False),
               HGroup(Item('distance', show_label=False, width=20),
                      'omit_points', 'reset_omit_points', show_labels=False),
               Item('omitted_info', style='readonly', show_label=False),
               label='Head Shape Source (Raw/Epochs/Evoked/DigMontage)',
               show_border=True, show_labels=False),
        show_labels=False, label="Data Source")

    # Setting `scrollable=True` for a Group does not seem to have any effect
    # (macOS), in order to be effective the parameter has to be set for a View
    # object; hence we use a special InstanceEditor to set the parameter
    # programmatically:
    coreg_panel = VGroup(
        # width=410 is optimized for macOS to avoid a horizontal scroll-bar;
        # might benefit from platform-specific values
        Item('coreg_panel', style='custom', width=410 if scrollable else 1,
             editor=InstanceEditor(view=_make_view_coreg_panel(scrollable))),
        label="Coregistration", show_border=not scrollable, show_labels=False,
        enabled_when="fid_panel.locked")

    main_layout = 'split' if split else 'normal'

    if tabbed:
        main = HGroup(scene,
                      Group(data_panel, coreg_panel, show_labels=False,
                            layout='tabbed'),
                      layout=main_layout)
    else:
        main = HGroup(data_panel, scene, coreg_panel, show_labels=False,
                      layout=main_layout)

    # Here we set the width and height to impossibly small numbers to force the
    # window to be as tight as possible
    view = View(main, resizable=True, handler=CoregFrameHandler(),
                buttons=NoButtons, width=scene_width, height=scene_height)
    return view


class ViewOptionsPanel(HasTraits):
    """View options panel."""

    mri_obj = Instance(SurfaceObject)
    hsp_obj = Instance(PointObject)
    eeg_obj = Instance(PointObject)
    hpi_obj = Instance(PointObject)
    view = View(VGroup(Item('mri_obj', style='custom',
                            label="MRI"),
                       Item('hsp_obj', style='custom',
                            label="Head shape"),
                       Item('eeg_obj', style='custom',
                            label='EEG'),
                       Item('hpi_obj', style='custom',
                            label='HPI'),
                       ), title="View Options")


class CoregFrame(HasTraits):
    """GUI for head-MRI coregistration."""

    model = Instance(CoregModel)

    scene = Instance(MlabSceneModel, ())
    headview = Instance(HeadViewController)

    subject_panel = Instance(SubjectSelectorPanel)
    fid_panel = Instance(FiducialsPanel)
    coreg_panel = Instance(CoregPanel)
    view_options_panel = Instance(ViewOptionsPanel)

    raw_src = DelegatesTo('model', 'hsp')
    guess_mri_subject = DelegatesTo('model')
    project_to_surface = DelegatesTo('eeg_obj')
    orient_to_surface = DelegatesTo('hsp_obj')
    scale_by_distance = DelegatesTo('hsp_obj')
    mark_inside = DelegatesTo('hsp_obj')

    # Omit Points
    distance = Float(5., desc="maximal distance for head shape points from "
                     "the surface (mm)")
    omit_points = Button(label='Omit [mm]', desc="to omit head shape points "
                         "for the purpose of the automatic coregistration "
                         "procedure.")
    reset_omit_points = Button(label='Reset', desc="to reset the "
                               "omission of head shape points to include all.")
    omitted_info = Property(Str, depends_on=['model:hsp:n_omitted'])

    fid_ok = DelegatesTo('model', 'mri.fid_ok')
    lock_fiducials = DelegatesTo('model')
    hsp_always_visible = Bool(False, label="Always Show Head Shape")
    title = Str('MNE Coreg')

    # visualization (MRI)
    mri_obj = Instance(SurfaceObject)
    mri_lpa_obj = Instance(PointObject)
    mri_nasion_obj = Instance(PointObject)
    mri_rpa_obj = Instance(PointObject)
    # visualization (Digitization)
    hsp_obj = Instance(PointObject)
    eeg_obj = Instance(PointObject)
    hpi_obj = Instance(PointObject)
    hsp_lpa_obj = Instance(PointObject)
    hsp_nasion_obj = Instance(PointObject)
    hsp_rpa_obj = Instance(PointObject)
    hsp_visible = Property(depends_on=['hsp_always_visible', 'lock_fiducials'])

    view_options = Button(label="View Options")

    picker = Instance(object)

    # Processing
    queue = DelegatesTo('coreg_panel')

    view = _make_view()

    def _model_default(self):
        return CoregModel(
            scale_labels=self._config.get(
                'MNE_COREG_SCALE_LABELS', 'true') == 'true',
            copy_annot=self._config.get(
                'MNE_COREG_COPY_ANNOT', 'true') == 'true',
            prepare_bem_model=self._config.get(
                'MNE_COREG_PREPARE_BEM', 'true') == 'true')

    def _subject_panel_default(self):
        return SubjectSelectorPanel(model=self.model.mri.subject_source)

    def _fid_panel_default(self):
        return FiducialsPanel(model=self.model.mri, headview=self.headview)

    def _coreg_panel_default(self):
        return CoregPanel(model=self.model)

    def _headview_default(self):
        return HeadViewController(scene=self.scene, system='RAS')

    def __init__(self, raw=None, subject=None, subjects_dir=None,
                 guess_mri_subject=True, head_opacity=1.,
                 head_high_res=True, trans=None, config=None,
                 project_eeg=False, orient_to_surface=False,
                 scale_by_distance=False, mark_inside=False):  # noqa: D102
        self._config = config or {}
        super(CoregFrame, self).__init__(guess_mri_subject=guess_mri_subject)
        self.model.mri.subject_source.show_high_res_head = head_high_res
        self._initial_kwargs = dict(project_eeg=project_eeg,
                                    orient_to_surface=orient_to_surface,
                                    scale_by_distance=scale_by_distance,
                                    mark_inside=mark_inside,
                                    head_opacity=head_opacity)
        if not 0 <= head_opacity <= 1:
            raise ValueError(
                "head_opacity needs to be a floating point number between 0 "
                "and 1, got %r" % (head_opacity,))

        if (subjects_dir is not None) and os.path.isdir(subjects_dir):
            self.model.mri.subjects_dir = subjects_dir

        if raw is not None:
            self.model.hsp.file = raw

        if subject is not None:
            if subject not in self.model.mri.subject_source.subjects:
                msg = "%s is not a valid subject. " % subject
                # no subjects -> ['']
                if any(self.model.mri.subject_source.subjects):
                    ss = ', '.join(self.model.mri.subject_source.subjects)
                    msg += ("The following subjects have been found: %s "
                            "(subjects_dir=%s). " %
                            (ss, self.model.mri.subjects_dir))
                else:
                    msg += ("No subjects were found in subjects_dir=%s. " %
                            self.model.mri.subjects_dir)
                msg += ("Make sure all MRI subjects have head shape files "
                        "(run $ mne make_scalp_surfaces).")
                raise ValueError(msg)
            self.model.mri.subject = subject
        if trans is not None:
            try:
                self.model.load_trans(trans)
            except Exception as e:
                error(None, "Error loading trans file %s: %s (See terminal "
                      "for details)" % (trans, e), "Error Loading Trans File")

    @on_trait_change('subject_panel:subject')
    def _set_title(self):
        self.title = '%s - MNE Coreg' % self.model.mri.subject

    @on_trait_change('scene:activated')
    def _init_plot(self):
        _toggle_mlab_render(self, False)

        lpa_color = defaults['lpa_color']
        nasion_color = defaults['nasion_color']
        rpa_color = defaults['rpa_color']

        # MRI scalp
        color = defaults['head_color']
        self.mri_obj = SurfaceObject(
            points=np.empty((0, 3)), color=color, tri=np.empty((0, 3)),
            scene=self.scene, name="MRI Scalp", block_behind=True,
            # opacity=self._initial_kwargs['head_opacity'],
            # setting opacity here causes points to be
            # [[0, 0, 0]] -- why??
        )
        self.mri_obj.opacity = self._initial_kwargs['head_opacity']
        self.fid_panel.hsp_obj = self.mri_obj
        # Do not do sync_trait here, instead use notifiers elsewhere

        # MRI Fiducials
        point_scale = defaults['mri_fid_scale']
        self.mri_lpa_obj = PointObject(scene=self.scene, color=lpa_color,
                                       point_scale=point_scale, name='LPA')
        self.model.sync_trait('transformed_mri_lpa',
                              self.mri_lpa_obj, 'points', mutual=False)
        self.mri_nasion_obj = PointObject(scene=self.scene, color=nasion_color,
                                          point_scale=point_scale,
                                          name='Nasion')
        self.model.sync_trait('transformed_mri_nasion',
                              self.mri_nasion_obj, 'points', mutual=False)
        self.mri_rpa_obj = PointObject(scene=self.scene, color=rpa_color,
                                       point_scale=point_scale, name='RPA')
        self.model.sync_trait('transformed_mri_rpa',
                              self.mri_rpa_obj, 'points', mutual=False)

        # Digitizer Head Shape
        kwargs = dict(
            view='cloud', scene=self.scene, resolution=20,
            orient_to_surface=self._initial_kwargs['orient_to_surface'],
            scale_by_distance=self._initial_kwargs['scale_by_distance'],
            mark_inside=self._initial_kwargs['mark_inside'])
        self.hsp_obj = PointObject(
            color=defaults['extra_color'], name='Extra',
            point_scale=defaults['extra_scale'], **kwargs)
        self.model.hsp.sync_trait('points', self.hsp_obj, mutual=False)

        # Digitizer EEG
        self.eeg_obj = PointObject(
            color=defaults['eeg_color'], point_scale=defaults['eeg_scale'],
            name='EEG', projectable=True,
            project_to_surface=self._initial_kwargs['project_eeg'], **kwargs)
        self.model.hsp.sync_trait('eeg_points', self.eeg_obj, 'points',
                                  mutual=False)

        # Digitizer HPI
        self.hpi_obj = PointObject(
            color=defaults['hpi_color'], name='HPI',
            point_scale=defaults['hpi_scale'], **kwargs)
        self.model.hsp.sync_trait('hpi_points', self.hpi_obj, 'points',
                                  mutual=False)
        for p in (self.hsp_obj, self.eeg_obj, self.hpi_obj):
            self.model.mri.bem_low_res.sync_trait('tris', p, 'project_to_tris',
                                                  mutual=False)
            self.model.sync_trait('transformed_low_res_mri_points',
                                  p, 'project_to_points', mutual=False)
            p.inside_color = self.mri_obj.color
            self.mri_obj.sync_trait('color', p, 'inside_color',
                                    mutual=False)

        # Digitizer Fiducials
        point_scale = defaults['dig_fid_scale']
        opacity = defaults['dig_fid_opacity']
        self.hsp_lpa_obj = PointObject(
            scene=self.scene, color=lpa_color, opacity=opacity,
            point_scale=point_scale, name='HSP-LPA')
        self.model.hsp.sync_trait('lpa', self.hsp_lpa_obj, 'points',
                                  mutual=False)
        self.hsp_nasion_obj = PointObject(
            scene=self.scene, color=nasion_color, opacity=opacity,
            point_scale=point_scale, name='HSP-Nasion')
        self.model.hsp.sync_trait('nasion', self.hsp_nasion_obj, 'points',
                                  mutual=False)
        self.hsp_rpa_obj = PointObject(
            scene=self.scene, color=rpa_color, opacity=opacity,
            point_scale=point_scale, name='HSP-RPA')
        self.model.hsp.sync_trait('rpa', self.hsp_rpa_obj, 'points',
                                  mutual=False)

        # All points share these
        for p in (self.hsp_obj, self.eeg_obj, self.hpi_obj,
                  self.hsp_lpa_obj, self.hsp_nasion_obj, self.hsp_rpa_obj):
            self.sync_trait('hsp_visible', p, 'visible', mutual=False)

        on_pick = self.scene.mayavi_scene.on_mouse_pick
        self.picker = on_pick(self.fid_panel._on_pick, type='cell')

        self.headview.left = True
        self._on_mri_src_change()
        _toggle_mlab_render(self, True)
        self.scene.render()
        self.scene.camera.focal_point = (0., 0., 0.)
        self.view_options_panel = ViewOptionsPanel(
            mri_obj=self.mri_obj, hsp_obj=self.hsp_obj, eeg_obj=self.eeg_obj,
            hpi_obj=self.hpi_obj)

    @cached_property
    def _get_hsp_visible(self):
        return self.hsp_always_visible or self.lock_fiducials

    @cached_property
    def _get_omitted_info(self):
        if self.model.hsp.n_omitted == 0:
            return "No points omitted"
        elif self.model.hsp.n_omitted == 1:
            return "1 point omitted"
        else:
            return "%i points omitted" % self.model.hsp.n_omitted

    def _omit_points_fired(self):
        distance = self.distance / 1000.
        self.model.omit_hsp_points(distance)

    def _reset_omit_points_fired(self):
        self.model.omit_hsp_points(0, True)

    @on_trait_change('model:transformed_high_res_mri_points')
    def _update_mri_obj_points(self):
        if self.mri_obj is None:
            return
        self.mri_obj.points = getattr(
            self.model, 'transformed_%s_res_mri_points'
            % ('high'
               if self.model.mri.subject_source.show_high_res_head else
               'low',))

    @on_trait_change('model:mri:bem_high_res.tris,'
                     'model:mri:subject_source:show_high_res_head')
    def _on_mri_src_change(self):
        if self.mri_obj is None:
            return
        if not (np.any(self.model.mri.bem_low_res.points) and
                np.any(self.model.mri.bem_low_res.tris)):
            self.mri_obj.clear()
            return

        if self.model.mri.subject_source.show_high_res_head:
            bem = self.model.mri.bem_high_res
        else:
            bem = self.model.mri.bem_low_res
        self.mri_obj.tri = bem.tris
        self._update_mri_obj_points()
        self.mri_obj.plot()

    # automatically lock fiducials if a good fiducials file is loaded
    @on_trait_change('model:mri:fid_file')
    def _on_fid_file_loaded(self):
        if self.model.mri.fid_file:
            self.fid_panel.locked = True
        else:
            self.fid_panel.locked = False

    def _view_options_fired(self):
        self.view_options_panel.edit_traits()

    def save_config(self, home_dir=None):
        """Write configuration values."""
        set_config('MNE_COREG_GUESS_MRI_SUBJECT',
                   str(self.model.guess_mri_subject).lower(),
                   home_dir, set_env=False)
        set_config(
            'MNE_COREG_HEAD_HIGH_RES',
            str(self.model.mri.subject_source.show_high_res_head).lower(),
            home_dir, set_env=False)
        set_config('MNE_COREG_HEAD_OPACITY',
                   str(self.mri_obj.opacity),
                   home_dir, set_env=False)
        # 'MNE_COREG_SCENE_WIDTH'
        # 'MNE_COREG_SCENE_HEIGHT'
        set_config('MNE_COREG_SCALE_LABELS',
                   str(self.model.scale_labels).lower(),
                   home_dir, set_env=False)
        set_config('MNE_COREG_COPY_ANNOT',
                   str(self.model.copy_annot).lower(),
                   home_dir, set_env=False)
        set_config('MNE_COREG_PREPARE_BEM',
                   str(self.model.prepare_bem_model).lower(),
                   home_dir, set_env=False)
        if self.model.mri.subjects_dir:
            set_config('MNE_COREG_SUBJECTS_DIR',
                       self.model.mri.subjects_dir,
                       home_dir, set_env=False)
        set_config('MNE_COREG_PROJECT_EEG',
                   str(self.project_to_surface).lower())
        set_config('MNE_COREG_ORIENT_TO_SURFACE',
                   str(self.orient_to_surface).lower())
        set_config('MNE_COREG_SCALE_BY_DISTANCE',
                   str(self.scale_by_distance).lower())
        set_config('MNE_COREG_MARK_INSIDE',
                   str(self.mark_inside).lower())
