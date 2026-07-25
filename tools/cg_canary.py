"""pytest plugin: detect CGColorSpace over-release, per test.

Arms buffer-retains on the CG colorspace objects Qt's cursor conversion can
pass to CGImageCreate (named sRGB, deviceRGB, and the ICC cache entry for the
"VP2030 SERIES" profile embedded in Qt's macOS wait/busy cursor PNGs), then
probes CFGetRetainCount after every test. A persistent drop names the test
that stole references. The buffer retains also *prevent* the SIGTRAP crash.

Use: PYTHONPATH=<dir> pytest -p cg_canary ...   (macOS only, no-op elsewhere)

Under pytest-xdist, worker stdout is redirected to /dev/null by execnet, so
set CG_CANARY_LOG=<path> to also append findings to a file (one shared file
is fine; lines are tagged with the pid).
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import sys

_armed = {}
_reported = set()


def _log(msg):
    line = f"[cg-canary] [{os.getpid()}] {msg}"
    print(line, flush=True)
    path = os.environ.get("CG_CANARY_LOG")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


if sys.platform == "darwin":
    import ctypes

    _cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    _cf = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    _cg.CGColorSpaceCreateWithName.restype = ctypes.c_void_p
    _cg.CGColorSpaceCreateWithName.argtypes = [ctypes.c_void_p]
    _cg.CGColorSpaceCreateDeviceRGB.restype = ctypes.c_void_p
    _cg.CGColorSpaceCreateWithICCData.restype = ctypes.c_void_p
    _cg.CGColorSpaceCreateWithICCData.argtypes = [ctypes.c_void_p]
    _cf.CFDataCreate.restype = ctypes.c_void_p
    _cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
    _cf.CFRetain.restype = ctypes.c_void_p
    _cf.CFRetain.argtypes = [ctypes.c_void_p]
    _cf.CFGetRetainCount.restype = ctypes.c_long
    _cf.CFGetRetainCount.argtypes = [ctypes.c_void_p]

    _IMMORTAL = (1 << 32) - 2
    _BUFFER = 500

    def _arm_one(label, ptr):
        if not ptr or label in _armed:
            return
        rc = _cf.CFGetRetainCount(ptr)
        if rc >= _IMMORTAL:
            _armed[label] = (ptr, None)  # immortal: track for info only
            _log(f"{label}: immortal, not tracked")
            return
        for _ in range(_BUFFER):
            _cf.CFRetain(ptr)
        base = _cf.CFGetRetainCount(ptr)
        _armed[label] = (ptr, base)
        _log(f"{label}: armed ptr={ptr:#x} rc={base}")

    def _try_arm():
        if "srgb-named" not in _armed:
            k = ctypes.c_void_p.in_dll(_cg, "kCGColorSpaceSRGB")
            _arm_one("srgb-named", _cg.CGColorSpaceCreateWithName(k))
        if "device-rgb" not in _armed:
            _arm_one("device-rgb", _cg.CGColorSpaceCreateDeviceRGB())
        if "vp2030-icc" not in _armed:
            # needs the cocoa plugin resources -> only after QApplication
            try:
                from qtpy.QtGui import QImage
                from qtpy.QtWidgets import QApplication

                if QApplication.instance() is None:
                    return
                img = QImage(":/qt-project.org/mac/cursors/images/spincursor.png")
                if img.isNull() or not img.colorSpace().isValid():
                    return
                icc = bytes(img.colorSpace().iccProfile().data())
                d = _cf.CFDataCreate(None, icc, len(icc))
                _arm_one("vp2030-icc", _cg.CGColorSpaceCreateWithICCData(d))
            except Exception:
                pass

    def pytest_runtest_teardown(item):
        _try_arm()
        for label, (ptr, base) in _armed.items():
            if base is None:
                continue
            rc = _cf.CFGetRetainCount(ptr)
            if rc != base and (label, rc) not in _reported:
                _reported.add((label, rc))
                _log(f"*** {label} rc {base} -> {rc} after {item.nodeid} ***")
                _armed[label] = (ptr, rc)  # new baseline; report each change

    def pytest_terminal_summary(terminalreporter):
        for label, (ptr, base) in _armed.items():
            state = "immortal" if base is None else f"final rc={base}"
            terminalreporter.write_line(f"[cg-canary] {label}: {state}")
