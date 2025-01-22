from setuptools import setup

setup(
    name="mujoco-xml-editor",
    version="0.0.0",
    packages=["mujoco_xml_editor"],
    install_requires=[
        "numpy",
        "coacd",
        "lxml",
        "scikit-robot",
    ],
    extras_require={
        "examples": [
            "mujoco",
            "mujoco-python-viewer",
            "ycb-utils",
            "robot_descriptions",
        ]
    },
    package_data={"mujoco": ["py.typed"]},
)
