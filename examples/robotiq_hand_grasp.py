import argparse
from pathlib import Path

import mujoco
import mujoco_viewer
import numpy as np
import ycb_utils
from robot_descriptions.robotiq_2f85_mj_description import PACKAGE_PATH

from mujoco_xml_editor import MujocoXmlEditor


def set_mocap_position(data: mujoco.MjData, pos: np.ndarray, hand_also: bool = False):
    pos = np.array(pos)
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([0.0, 0.5 * np.pi, 0.5 * np.pi]), "xyz")
    if hand_also:
        data.qpos[0:3] = pos
        data.qpos[3:7] = quat
    data.mocap_pos[0] = pos
    data.mocap_quat[0] = np.array(quat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize")
    args = parser.parse_args()

    hand_xml_path = Path(PACKAGE_PATH) / "2f85.xml"
    editor = MujocoXmlEditor.load(hand_xml_path)
    mesh_path = ycb_utils.resolve_path("019_pitcher_base")
    editor.add_mesh(
        mesh_path, "pitcher_base", convex_decomposition=True, pos=[0.3, 0, 0], euler=[0, 0, 2.3704]
    )
    editor.add_ground()
    editor.add_sky()
    editor.add_light()
    editor.add_mocap("mocap", "base_mount")  # base_mount is the root body of the robotiq hand
    xmlstr = editor.to_string()

    model = mujoco.MjModel.from_xml_string(xmlstr)
    data = mujoco.MjData(model)

    if args.visualize:
        viewer = mujoco_viewer.MujocoViewer(model, data)

        finger_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

        input("press enter to start the simulation")
        pos = np.array([0.0, -0.02, 0.15])
        # very important that the hand is also moved to the same position
        set_mocap_position(data, pos, hand_also=True)

        # move the finger down
        for i in range(110):
            pos[0] += 0.0004
            set_mocap_position(data, pos)
            data.ctrl[finger_act_id] = 100
            mujoco.mj_step(model, data)
            viewer.render()
        print("done1")

        # grasp
        for i in range(300):
            data.ctrl[finger_act_id] = 220
            mujoco.mj_step(model, data)
            viewer.render()
        print("done2")

        # move the finger up
        for i in range(300):
            pos[2] += 0.0004
            set_mocap_position(data, pos)
            mujoco.mj_step(model, data)
            viewer.render()

        while True:
            mujoco.mj_step(model, data)
            viewer.render()
