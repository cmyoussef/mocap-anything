# ------------------------------------------------------------------------
# FACTORY FUNCTION
# ------------------------------------------------------------------------
try:
    import maya.cmds as cmds

    MAYA_AVAILABLE = True
except ImportError:
    cmds = None
    MAYA_AVAILABLE = False
    print("[INFO] Maya not available in this environment. Maya-specific functions will no-op.")

from QuadCap.handlers.skeleton_builder import SkeletonBuilderBase

SkeletonBuilder = SkeletonBuilderBase
if MAYA_AVAILABLE:
    from QuadCap.handlers.maya_handler.maya_skeleton_builder import MayaSkeletonBuilder

    SkeletonBuilder = MayaSkeletonBuilder
