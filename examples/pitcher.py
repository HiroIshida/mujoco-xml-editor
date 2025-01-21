from xml.etree.ElementTree import tostring
from lxml import etree
import ycb_utils
from mujoco_xml_utils.xml_utils import _convex_decomposition, create_empty_xml, add_mesh

mesh_path = ycb_utils.resolve_path("019_pitcher_base")
xml = create_empty_xml(mesh_path, "example")
add_mesh(xml, mesh_path, "pitcher_base", convex_decomposition=True)
xmlstr = tostring(xml, encoding="utf8")
et = etree.fromstring(xmlstr)
pretty_xml = etree.tostring(et, pretty_print=True, encoding=str)
print(pretty_xml)
