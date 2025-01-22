## mujoco-xml-utils
Utilities for working with MuJoCo XML files.

Install with:
```bash
sudo apt-get update
sudo apt-get install libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev
pip install .[examples] -v
```

Write mujoco xml as follows:
```python
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
```
