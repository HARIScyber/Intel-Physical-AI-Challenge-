"""MuJoCo XML construction for the dinner-table task."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def build_scene_xml(config: dict[str, Any]) -> str:
    """Build a self-contained articulated MuJoCo scene.

    The arms use real MuJoCo rigid bodies, hinge joints, transmissions, and
    contact geoms. The XML is intentionally generated so dimensions and the
    scene remain configuration-relative and can later be replaced by an
    official SO-101 MJCF asset.
    """
    table = config.get("objects", {}).get("table_size", [1.6, 1.0, 0.75])
    try:
        length, width, height = (float(value) for value in table)
    except (TypeError, ValueError) as exc:
        LOGGER.exception("Invalid table_size configuration")
        raise ValueError("objects.table_size must contain three numbers") from exc
    timestep = float(config.get("simulation", {}).get("timestep", 0.002))
    return f'''<mujoco model="so101_dinner_table">
  <compiler angle="radian" inertiafromgeom="true" autolimits="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <size njmax="2000" nconmax="500"/>
  <visual><headlight diffuse="0.8 0.8 0.8" ambient="0.35 0.35 0.35"/></visual>
  <default>
    <joint damping="0.8" armature="0.01"/>
    <geom friction="0.8 0.1 0.02" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    <motor ctrlrange="-1 1"/>
  </default>
  <actuator>
    {_actuator_xml("a")}
    {_actuator_xml("b")}
    <position name="drawer_motor" joint="drawer_slide" kp="80" ctrlrange="-0.35 0.02"/>
  </actuator>
  <contact>
    <exclude body1="arm_a_gripper_left" body2="arm_a_gripper_right"/>
    <exclude body1="arm_b_gripper_left" body2="arm_b_gripper_right"/>
    <exclude body1="arm_a_link_4" body2="arm_a_gripper_left"/>
    <exclude body1="arm_a_link_4" body2="arm_a_gripper_right"/>
    <exclude body1="arm_b_link_4" body2="arm_b_gripper_left"/>
    <exclude body1="arm_b_link_4" body2="arm_b_gripper_right"/>
  </contact>
  <worldbody>
    <light name="key_light" pos="0 -1.5 3" dir="0 0 -1" diffuse="0.9 0.85 0.75" castshadow="true"/>
    <light name="fill_light" pos="0 1.5 2" dir="0 0 -1" diffuse="0.35 0.45 0.6"/>
    <camera name="overhead" pos="0 -2.5 3.2" xyaxes="1 0 0 0 0.8 0.6"/>
    <camera name="front" pos="0 -3.0 1.6" xyaxes="1 0 0 0 0.45 -0.89"/>
    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.12 0.15 0.18 1" contype="1" conaffinity="1"/>
    <body name="table" pos="0 0 {height / 2}">
      <geom name="tabletop" type="box" size="{length / 2} {width / 2} {height / 2}" rgba="0.45 0.24 0.12 1"/>
      <geom name="table_front" type="box" pos="0 {-width / 2 - 0.025} 0" size="{length / 2} 0.025 {height / 2}" rgba="0.35 0.17 0.08 1"/>
      <body name="drawer" pos="0 {-width / 2 - 0.07} 0.18">
        <joint name="drawer_slide" type="slide" axis="0 -1 0" range="-0.35 0.02" damping="2"/>
        <geom name="drawer_box" type="box" size="0.42 0.28 0.13" rgba="0.55 0.30 0.14 1"/>
        <geom name="drawer_handle" type="box" pos="0 -0.30 0" size="0.13 0.025 0.025" rgba="0.12 0.12 0.12 1"/>
      </body>
    </body>
    <body name="plate" pos="-0.25 0 0.768">
      <joint name="plate_free" type="free"/>
      <geom name="plate_geom" type="cylinder" size="0.16 0.018" mass="0.18" rgba="0.92 0.92 0.86 1"/>
    </body>
    <body name="cup" pos="0.28 0.10 0.825">
      <joint name="cup_free" type="free"/>
      <geom name="cup_geom" type="cylinder" size="0.055 0.075" mass="0.12" rgba="0.12 0.38 0.80 1"/>
    </body>
    <body name="spoon" pos="-0.20 -0.18 0.762">
      <joint name="spoon_free" type="free"/>
      <geom name="spoon_handle" type="capsule" fromto="0 -0.09 0 0 0.09 0" size="0.012" mass="0.025" rgba="0.72 0.74 0.78 1"/>
      <geom name="spoon_bowl" type="ellipsoid" pos="0 0.105 0" size="0.035 0.05 0.008" mass="0.02" rgba="0.72 0.74 0.78 1"/>
    </body>
    <body name="fork" pos="-0.25 -0.20 0.762">
      <joint name="fork_free" type="free"/>
      <geom name="fork_handle" type="capsule" fromto="0 -0.10 0 0 0.10 0" size="0.012" mass="0.025" rgba="0.72 0.74 0.78 1"/>
      <geom name="fork_head" type="box" pos="0 0.115 0" size="0.035 0.035 0.008" mass="0.02" rgba="0.72 0.74 0.78 1"/>
    </body>
    <body name="napkin" pos="-0.30 -0.18 0.756"><joint name="napkin_free" type="free"/><geom type="box" size="0.12 0.09 0.006" mass="0.01" rgba="0.85 0.18 0.16 1"/></body>
    <body name="bowl" pos="0.45 0.18 0.795"><joint name="bowl_free" type="free"/><geom type="cylinder" size="0.12 0.045" mass="0.14" rgba="0.88 0.68 0.25 1"/></body>
    { _arm_xml("a", -0.92, 0.05, "0.15 0.55 0.42 1") }
    { _arm_xml("b", 0.92, 0.05, "0.75 0.28 0.18 1") }
  </worldbody>
</mujoco>'''


def _arm_xml(label: str, x: float, y: float, rgba: str) -> str:
    """Return a six-joint articulated arm and two physical gripper fingers."""
    side = -1.0 if label == "a" else 1.0
    return f'''<body name="so101_{label}_base" pos="{x} {y} 0.8">
      <joint name="arm_{label}_joint_1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom name="arm_{label}_base_geom" type="cylinder" size="0.12 0.08" rgba="{rgba}"/>
      <body name="arm_{label}_link_1" pos="0 0 0.16">
        <joint name="arm_{label}_joint_2" type="hinge" axis="0 1 0" range="-1.7 1.7"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.07" rgba="{rgba}"/>
        <body name="arm_{label}_link_2" pos="0 0 0.30">
          <joint name="arm_{label}_joint_3" type="hinge" axis="0 1 0" range="-2.4 2.4"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.28" size="0.06" rgba="{rgba}"/>
          <body name="arm_{label}_link_3" pos="0 0 0.28">
            <joint name="arm_{label}_joint_4" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.20" size="0.05" rgba="{rgba}"/>
            <body name="arm_{label}_link_4" pos="0 0 0.20">
              <joint name="arm_{label}_joint_5" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.14" size="0.045" rgba="{rgba}"/>
              <body name="arm_{label}_wrist" pos="0 0 0.14">
                <joint name="arm_{label}_joint_6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom name="arm_{label}_wrist_geom" type="capsule" fromto="0 0 0 0 0 0.08" size="0.04" rgba="{rgba}"/>
                <site name="arm_{label}_grasp_site" pos="0 0 {0.10 * side}" size="0.025" rgba="1 0 0 1"/>
                <body name="arm_{label}_gripper_left" pos="0.035 0 0.08"><joint name="arm_{label}_gripper_left_slide" type="slide" axis="1 0 0" range="-0.02 0.02"/><geom type="box" size="0.015 0.035 0.06" rgba="{rgba}"/></body>
                <body name="arm_{label}_gripper_right" pos="-0.035 0 0.08"><joint name="arm_{label}_gripper_right_slide" type="slide" axis="1 0 0" range="-0.02 0.02"/><geom type="box" size="0.015 0.035 0.06" rgba="{rgba}"/></body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>'''
def _actuator_xml(label: str) -> str:
    """Return position actuators for six joints and the two gripper slides."""
    joints = "\n".join(
        f'    <position name="arm_{label}_motor_{index}" joint="arm_{label}_joint_{index}" kp="35" ctrlrange="-3.14 3.14"/>'
        for index in range(1, 7)
    )
    return joints + f'''
    <position name="arm_{label}_gripper_left_motor" joint="arm_{label}_gripper_left_slide" kp="20" ctrlrange="-0.02 0.02"/>
    <position name="arm_{label}_gripper_right_motor" joint="arm_{label}_gripper_right_slide" kp="20" ctrlrange="-0.02 0.02"/>'''
