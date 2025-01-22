import argparse

import mujoco
import mujoco_viewer
from skrobot.model.primitives import Box, Cylinder

from mujoco_xml_editor import MujocoXmlEditor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    box = Box([1, 1, 1])
    box.translate([0, 0, 0.5])
    cylinder = Cylinder(0.1, 1)
    cylinder.translate([1.3, 0, 0.5])
    editor = MujocoXmlEditor.empty("test")
    editor.add_primitive(box, "box")
    editor.add_primitive(cylinder, "cylinder")
    editor.add_ground()
    editor.add_sky()
    editor.add_light()
    xmlstr = editor.to_string()

    model = mujoco.MjModel.from_xml_string(xmlstr)
    data = mujoco.MjData(model)

    if args.visualize:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        while True:
            mujoco.mj_step(model, data)
            viewer.render()
