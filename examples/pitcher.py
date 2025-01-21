import argparse

import mujoco
import mujoco_viewer
import numpy as np
import ycb_utils

from mujoco_xml_editor import MujocoXmlEditor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    editor = MujocoXmlEditor.empty("example")
    mesh_path = ycb_utils.resolve_path("019_pitcher_base")
    editor.add_mesh(mesh_path, "pitcher_base", convex_decomposition=True)
    editor.add_ground()
    editor.add_sky()
    editor.add_light()
    editor.add_mocap("mocap", "pitcher_base")
    xmlstr = editor.to_string()

    model = mujoco.MjModel.from_xml_string(xmlstr)
    data = mujoco.MjData(model)

    if args.visualize:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        pos = np.array([0.0, 0.0, 0.0])

        while True:
            mujoco.mj_step(model, data)
            viewer.render()
            pos[2] += 0.001
            data.mocap_pos[0] = pos
