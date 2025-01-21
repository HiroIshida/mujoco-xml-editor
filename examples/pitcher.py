import ycb_utils

from mujoco_xml_utils.xml_utils import MujocoXmlEditor

editor = MujocoXmlEditor.empty("example")
mesh_path = ycb_utils.resolve_path("019_pitcher_base")
editor.add_mesh(mesh_path, "pitcher_base", convex_decomposition=True)
editor.add_ground()
editor.add_sky()
editor.add_light()
xmlstr = editor.to_string()
print(xmlstr)
