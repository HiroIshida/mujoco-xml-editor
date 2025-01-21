from hashlib import md5
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, parse, tostring

import coacd
import numpy as np
import trimesh
from lxml import etree
from trimesh import Trimesh


class MujocoXmlEditor:
    def __init__(self, root: Element):
        self.root = root

    @classmethod
    def empty(cls, model_name: str):
        root = Element("mujoco")
        root.set("model", model_name)
        return cls(root)

    @classmethod
    def load(cls, file_path: Path):
        root = parse(file_path).getroot()
        asset = root.find("asset")

        # fix relative paths
        if asset is not None:
            package_path = file_path.parent
            leaf_files = list(package_path.rglob("*"))
            for mesh in asset.findall("mesh"):
                relative_path = Path(mesh.get("file"))
                for leaf_file in leaf_files:
                    if leaf_file.name == relative_path.name:
                        mesh.set("file", str(leaf_file.expanduser()))
                        break
        return cls(root)

    def to_string(self) -> str:
        xmlstr = tostring(self.root, encoding="utf8")
        et = etree.fromstring(xmlstr)
        pretty_xml = etree.tostring(et, pretty_print=True, encoding=str)
        return pretty_xml

    def add_mesh(
        self,
        mesh_file: Path,
        name: str,
        convex_decomposition: bool = True,
        cd_threshold=0.02,
        pos: np.ndarray = np.zeros(3),
        euler: np.ndarray = np.zeros(3),
    ) -> None:
        # create asset
        asset = self._create_element_if_not_exists(self.root, "asset")
        if convex_decomposition:
            mesh = trimesh.load(mesh_file)
            cache_path = self._convex_decomposition(mesh, cd_threshold)
            assert len(list(cache_path.iterdir())) > 0
            for i, part_file in enumerate(cache_path.iterdir()):
                part_name = f"{name}_part_{i}"
                mesh = SubElement(asset, "mesh", {"file": str(part_file), "name": part_name})
        else:
            mesh = SubElement(asset, "mesh", {"file": str(mesh_file), "name": name})

        # create body
        worldbody = self._create_element_if_not_exists(self.root, "worldbody")
        body = SubElement(
            worldbody,
            "body",
            attrib={
                "name": name,
                "pos": " ".join(map(str, pos)),
                "euler": " ".join(map(str, euler)),
            },
        )
        SubElement(body, "joint", attrib={"name": "joint_" + name, "type": "free"})
        if convex_decomposition:
            for i in range(len(list(cache_path.iterdir()))):
                part_name = f"{name}_part_{i}"
                geom = SubElement(
                    body, "geom", attrib={"mesh": part_name, "type": "mesh", "density": "1000"}
                )
        else:
            geom = SubElement(
                body, "geom", attrib={"mesh": name, "type": "mesh", "density": "1000"}
            )

    def add_mocap(
        self,
        name: str,
        attached_to: str,
        pos: np.ndarray = np.zeros(3),
        euler: np.ndarray = np.zeros(3),
    ):
        worldbody = self._create_element_if_not_exists(self.root, "worldbody")
        body = SubElement(
            worldbody,
            "body",
            attrib={
                "name": name,
                "pos": " ".join(map(str, pos)),
                "euler": " ".join(map(str, euler)),
                "mocap": "true",
            },
        )

        # create equality
        equality = self._create_element_if_not_exists(self.root, "equality")
        equality.append(Element("weld", {"body1": name, "body2": attached_to}))

        # check if attached to body exists
        body_elem = None
        for body in worldbody.findall("body"):
            if body.get("name") == attached_to:
                body_elem = body
                break
        assert body_elem is not None, f"Body {attached_to} does not exist"

        # add free joint
        body_elem_joint = body_elem.find("joint")
        if body_elem_joint is not None:
            assert (
                body_elem_joint.get("type") == "free"
            ), f"Joint {body_elem_joint.get('name')} is not free"
        else:
            SubElement(body_elem, "joint", attrib={"type": "free"})

    def add_ground(self):
        asset = self._create_element_if_not_exists(self.root, "asset")
        asset.append(
            Element(
                "texture",
                {
                    "type": "2d",
                    "name": "groundplane",
                    "builtin": "checker",
                    "mark": "edge",
                    "rgb1": "0.2 0.3 0.4",
                    "rgb2": "0.1 0.2 0.3",
                    "markrgb": "0.8 0.8 0.8",
                    "width": "300",
                    "height": "300",
                },
            )
        )
        asset.append(
            Element(
                "material",
                {
                    "name": "groundplane",
                    "texture": "groundplane",
                    "texuniform": "true",
                    "texrepeat": "5 5",
                    "reflectance": "0.2",
                },
            )
        )
        worldbody = self._create_element_if_not_exists(self.root, "worldbody")
        worldbody.append(
            Element(
                "geom",
                {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"},
            )
        )

    def add_sky(self):
        asset = self._create_element_if_not_exists(self.root, "asset")
        asset.append(
            Element(
                "texture",
                {
                    "type": "skybox",
                    "builtin": "gradient",
                    "rgb1": "0.3 0.5 0.7",
                    "rgb2": "0 0 0",
                    "width": "512",
                    "height": "3072",
                },
            )
        )

    def add_light(self):
        visual = self._create_element_if_not_exists(self.root, "visual")
        visual.append(
            Element(
                "headlight",
                {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
            )
        )
        visual.append(Element("rgba", {"haze": "0.15 0.25 0.35 1"}))
        visual.append(Element("global", {"azimuth": "150", "elevation": "-20"}))

    @staticmethod
    def _mesh_hash(mesh: Trimesh):
        return md5(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()

    @staticmethod
    def _convex_decomposition(mesh: Trimesh, threshold: float) -> Path:
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        dir_name = MujocoXmlEditor._mesh_hash(mesh) + f"threshold_{threshold}"
        cache_base_path = (
            Path("~/.cache/mujoco_xml_utils/convex_decomposition").expanduser() / dir_name
        )
        if not cache_base_path.exists():
            cache_base_path.mkdir(parents=True, exist_ok=True)
            parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
            for i, (V, F) in enumerate(parts):
                part = Trimesh(V, F)
                file_path = cache_base_path / f"part_{i}.obj"
                part.export(file_path)
        return cache_base_path

    @staticmethod
    def _create_element_if_not_exists(parent: Element, tag: str) -> Element:
        element = parent.find(tag)
        if element is None:
            element = SubElement(parent, tag)
        return element
