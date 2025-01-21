import trimesh
import numpy as np
from pathlib import Path
from trimesh import Trimesh
from hashlib import md5
import coacd
from xml.etree.ElementTree import Element, SubElement, tostring


def _mesh_hash(mesh: Trimesh):
    return md5(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest()


def _convex_decomposition(mesh: Trimesh, threshold: float) -> Path:
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    dir_name = _mesh_hash(mesh) + f"threshold_{threshold}"
    cache_base_path = Path("~/.cache/mujoco_xml_utils/convex_decomposition").expanduser() / dir_name
    if not cache_base_path.exists():
        cache_base_path.mkdir(parents=True, exist_ok=True)
        parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
        for i, (V, F) in enumerate(parts):
            part = Trimesh(V, F)
            file_path = cache_base_path / f"part_{i}.obj"
            part.export(file_path)
    return cache_base_path


def _create_element_if_not_exists(root: Element, tag: str) -> Element:
    element = root.find(tag)
    if element is None:
        element = SubElement(root, tag)
    return element


def add_mesh(
    root: Element,
    mesh_file: Path,
    name: str,
    convex_decomposition: bool = True,
    cd_threshold=0.02,
    pos: np.ndarray = np.zeros(3),
    euler: np.ndarray = np.zeros(3),
) -> None:
    # create asset
    asset = _create_element_if_not_exists(root, "asset")
    if convex_decomposition:
        mesh = trimesh.load(mesh_file)
        cache_path = _convex_decomposition(mesh, cd_threshold)
        assert len(list(cache_path.iterdir())) > 0
        for i, part_file in enumerate(cache_path.iterdir()):
            part_name = f"{name}_part_{i}"
            mesh = SubElement(asset, "mesh", {"file": str(part_file), "name": part_name})
    else:
        mesh = SubElement(asset, "mesh", {"file": str(mesh_file), "name": name})

    # create body
    worldbody = _create_element_if_not_exists(root, "worldbody")
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
        geom = SubElement(body, "geom", attrib={"mesh": name, "type": "mesh", "density": "1000"})


def create_empty_xml(file_path: Path, model_name: str) -> Element:
    root = Element("mujoco")
    root.set("model", model_name)
    return root
