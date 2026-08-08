"""Build one URDF articulation containing the UR10e and actual RG6 hand."""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
UR10E_SOURCE = ROOT / "assets" / "robots" / "ur10e" / "ur10e_lula_source.urdf"
RG6_SOURCE = ROOT / "assets" / "robots" / "onrobot_rg6" / "onrobot_rg6.urdf"
OUTPUT = ROOT / "assets" / "robots" / "ur10e_rg6" / "ur10e_rg6.urdf"
RG6_PREFIX = "rg6_"


def remove_children(root: ET.Element, tags: set[str]) -> None:
    for child in list(root):
        if child.tag in tags:
            root.remove(child)


def rewrite_ur10e_meshes(root: ET.Element) -> None:
    prefix = "package://ur_description/meshes/ur10e/"
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(prefix):
            mesh.set("filename", "../ur10e/meshes/" + filename[len(prefix) :])


def prefix_rg6_tree(element: ET.Element) -> ET.Element:
    cloned = copy.deepcopy(element)
    if cloned.tag in {"link", "joint"} and cloned.get("name"):
        cloned.set("name", RG6_PREFIX + cloned.get("name", ""))
    for descendant in cloned.iter():
        if descendant.tag in {"parent", "child"} and descendant.get("link"):
            descendant.set("link", RG6_PREFIX + descendant.get("link", ""))
        if descendant.tag == "mimic" and descendant.get("joint"):
            descendant.set("joint", RG6_PREFIX + descendant.get("joint", ""))
        if descendant.tag == "material" and descendant.get("name"):
            descendant.set("name", RG6_PREFIX + descendant.get("name", ""))
        if descendant.tag == "mesh":
            filename = descendant.get("filename", "")
            if filename.startswith("meshes/"):
                descendant.set(
                    "filename",
                    "../onrobot_rg6/" + filename,
                )
    return cloned


def validate(root: ET.Element) -> None:
    link_names = [element.get("name", "") for element in root.findall("link")]
    joint_names = [element.get("name", "") for element in root.findall("joint")]
    if len(link_names) != len(set(link_names)):
        raise ValueError("Composite URDF contains duplicate link names")
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("Composite URDF contains duplicate joint names")

    link_set = set(link_names)
    child_links: set[str] = set()
    moving_joints = 0
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.get('name')} has no parent/child")
        parent_name = parent.get("link", "")
        child_name = child.get("link", "")
        if parent_name not in link_set or child_name not in link_set:
            raise ValueError(
                f"Joint {joint.get('name')} references an unknown link"
            )
        child_links.add(child_name)
        if joint.get("type") != "fixed":
            moving_joints += 1

    roots = sorted(link_set - child_links)
    if roots != ["base_link"]:
        raise ValueError(f"Expected one base_link root, got {roots}")
    if moving_joints != 12:
        raise ValueError(f"Expected 12 moving joints, got {moving_joints}")

    missing_meshes = []
    for mesh in root.iter("mesh"):
        path = (OUTPUT.parent / mesh.get("filename", "")).resolve()
        if not path.is_file():
            missing_meshes.append(str(path))
    if missing_meshes:
        raise FileNotFoundError(f"Missing composite meshes: {missing_meshes}")


def main() -> None:
    ur10e_root = ET.parse(UR10E_SOURCE).getroot()
    rg6_root = ET.parse(RG6_SOURCE).getroot()

    remove_children(ur10e_root, {"ros2_control", "transmission", "gazebo"})
    rewrite_ur10e_meshes(ur10e_root)
    ur10e_root.set("name", "ur10e_rg6")

    for child in rg6_root:
        if child.tag in {"link", "joint"}:
            ur10e_root.append(prefix_rg6_tree(child))

    mount = ET.SubElement(
        ur10e_root,
        "joint",
        {"name": "rg6_mount_joint", "type": "fixed"},
    )
    ET.SubElement(mount, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(mount, "parent", {"link": "flange"})
    ET.SubElement(
        mount,
        "child",
        {"link": f"{RG6_PREFIX}onrobot_rg6_base_link"},
    )

    validate(ur10e_root)
    ET.indent(ur10e_root, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ur10e_root).write(
        OUTPUT,
        encoding="utf-8",
        xml_declaration=True,
    )
    print(f"UR10E_RG6_COMPOSITE_URDF={OUTPUT}")


if __name__ == "__main__":
    main()
