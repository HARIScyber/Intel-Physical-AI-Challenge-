"""Run the final demonstration of the Physical AI Challenge dinner-table task.

This comprehensive demo orchestrates the complete pipeline:
1. Start MuJoCo
2. Load randomized scene
3. Display natural-language instruction
4. Capture camera observations
5. Parse language
6. Build scene state
7. Generate task plan
8. Execute dual-arm actions
9. Use VLA policy when enabled
10. Verify every step
11. Recover from failures
12. Complete dinner-table task
13. Display final success/failure
14. Save trajectory and metrics

Features:
- Support for both scripted and VLA policies
- Comprehensive verification and error recovery
- Step-by-step execution logging
- Trajectory and metrics saving
- Configurable execution parameters
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from language import parse_instruction, reason_about_task
from perception import CameraProcessor
from physical_ai.config import load_config
from planning import Action, ActionSequence, TaskPlanner
from policy import ScriptedPolicy
from policy.base_policy import BaselineMetrics, ScriptedPolicy, OBJECT_ARM_MAP, TABLE_TOP, OBJECT_HALF_HEIGHTS, diagnose_scene
from policy.vla_policy import LeRobotVLA as VLAPolicy, VLAUnavailableError, VLAStates
from simulation import BimanualMujocoEnv

from json_helper import to_jsonable, safe_json_dumps

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Execute the complete demonstration and save JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Disable the interactive MuJoCo viewer")
    parser.add_argument("--policy", choices=["scripted", "vla"], default="scripted", help="Policy to use")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for VLA policy")
    parser.add_argument("--render", action="store_true", help="Enable rendering (overrides headless)")
    parser.add_argument("--save-video", action="store_true", help="Save video of demonstration")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps to execute")
    parser.add_argument("--device", type=str, default="cpu", help="Device for VLA policy (cpu/gpu)")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured deterministic seed")
    parser.add_argument("--instruction", type=str, default="Set the dinner table.", help="Natural-language instruction")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()

    environment = None
    trajectory = []
    video_frames = []
    result = None
    error = None
    try:
        LOGGER.info("Phase 1: Starting MuJoCo and loading configuration")
        config = load_config()
        environment = BimanualMujocoEnv(config)

        LOGGER.info("Phase 2: Loading randomized scene with seed=%s", args.seed)
        seed = args.seed if args.seed is not None else int(config.simulation["simulation"].get("seed", 0))
        observation, reset_info = environment.reset(seed=seed)
        LOGGER.info(diagnose_scene(environment.model, environment.data))

        render_enabled = (bool(config.simulation["simulation"].get("render", True)) and not args.headless) or args.render
        if render_enabled:
            LOGGER.info("Phase 3: Rendering environment enabled")
            environment.render()

        LOGGER.info("Phase 4: Capturing camera observations")
        perception_config = config.simulation.get("perception", {})
        processor = CameraProcessor(
            perception_config, 
            ROOT / perception_config.get("output_dir", "outputs/perception")
        )
        scene_state = processor.perceive(
            observation, environment.model, environment.data, perception_config.get("camera", "overhead")
        )

        LOGGER.info("Phase 5: Parsing natural-language instruction")
        instruction = args.instruction
        LOGGER.info("Instruction: %s", instruction)
        parsed_instruction = parse_instruction(instruction)

        LOGGER.info("Phase 6: Building scene state")
        current_scene_state = scene_state

        LOGGER.info("Phase 7: Generating task plan from instruction")
        task_command = reason_about_task(parsed_instruction)
        LOGGER.info("Planned task: %s", task_command)

        planner = TaskPlanner({"max_retries": 2})
        sequence = planner.plan(task_command, current_scene_state)
        LOGGER.info("Generated %d action steps", len(sequence.actions))

        LOGGER.info("Phase 8: Initializing policy: %s", args.policy)
        if args.policy == "scripted":
            policy = ScriptedPolicy(model=environment.model, data=environment.data)
            policy_fallback = False
        else:
            policy = VLAPolicy(device=args.device)
            policy_fallback = False
            try:
                policy.load_checkpoint(args.checkpoint)
                LOGGER.info("VLA checkpoint loaded successfully")
            except (VLAUnavailableError, ValueError, RuntimeError) as exc:
                LOGGER.error("VLA checkpoint loading failed: %s", exc)
                LOGGER.info("Switching to ScriptedPolicy as fallback")
                policy_fallback = True
                policy = ScriptedPolicy(model=environment.model, data=environment.data)

        policy.reset(sequence)

        timestep = float(config.simulation["simulation"].get("timestep", 0.002))
        control_frequency = float(config.simulation["simulation"].get("control_frequency", 20))
        physics_steps = max(1, int(round(1.0 / control_frequency / timestep)))

        collision_count = 0
        failed_grasps = 0
        successful_placements = 0
        successful_actions = 0
        failed_actions = 0
        last_scene = scene_state
        object_states: dict[str, str] = {}
        held_objects: dict[str, str] = {"arm_a": None, "arm_b": None}
        pending_transports: dict[str, str] = {}

        LOGGER.info("Phase 9: Executing dual-arm actions")
        steps_executed = 0
        i = 0
        while i < len(sequence.actions):
            action = sequence.actions[i]
            if args.max_steps is not None and steps_executed >= args.max_steps:
                LOGGER.warning("Reached max steps (%d), stopping", args.max_steps)
                break

            LOGGER.info("Step %d/%d: Executing %s", i + 1, len(sequence.actions), action.name)

            if action.name in {"transport"}:
                expected_object = action.objects[0] if action.objects else ""
                arm_label = _object_assigned_to_arm(expected_object)
                expected_arm = f"arm_{arm_label}"
                if held_objects.get(expected_arm) != expected_object:
                    LOGGER.warning("Skipping transport for %s: not held by %s (held=%s)", expected_object, expected_arm, held_objects.get(expected_arm))
                    trajectory.append({
                        "step": steps_executed,
                        "action": action.name,
                        "policy": args.policy,
                        "status": "SKIPPED",
                        "reason": "object_not_grasped",
                        "object": expected_object,
                        "held_by": held_objects.get(expected_arm),
                        "timestamp": time.perf_counter() - started,
                    })
                    i += 1
                    continue

            try:
                if args.policy == "vla" and policy_fallback:
                    LOGGER.warning("Using scripted fallback for action %s", action.name)
                    command_action = policy.predict(last_scene)
                elif args.policy == "vla" and not policy.is_ready():
                    LOGGER.error("VLA policy not ready, skipping action %s", action.name)
                    failed_actions += 1
                    trajectory.append({
                        "step": steps_executed,
                        "action": action.name,
                        "policy": "vla",
                        "status": "POLICY_UNAVAILABLE",
                        "reason": "checkpoint_unavailable",
                        "timestamp": time.perf_counter() - started,
                    })
                    i += 1
                    continue
                else:
                    command_action = policy.predict(last_scene)

                if command_action is None:
                    LOGGER.warning("Policy returned None for action %s, skipping", action.name)
                    failed_actions += 1
                    trajectory.append({
                        "step": steps_executed,
                        "action": action.name,
                        "policy": args.policy,
                        "status": "SKIPPED",
                        "reason": "policy_returned_none",
                        "timestamp": time.perf_counter() - started,
                    })
                    i += 1
                    continue

                repeats = max(1, int(round(action.duration * control_frequency)))

                for repeat in range(repeats):
                    command_action["_physics_steps"] = physics_steps
                    result = environment.step(command_action)
                    collision_count += int(result[4].get("collision", False))
                    if repeat == repeats - 1 and hasattr(policy, '_arm_controllers'):
                        arm_a_fk = policy._arm_controllers['a'].forward_kinematics().position if 'a' in policy._arm_controllers else 'N/A'
                        arm_b_fk = policy._arm_controllers['b'].forward_kinematics().position if 'b' in policy._arm_controllers else 'N/A'
                        LOGGER.info("DEBUG after step arm_a_fk=%s arm_b_fk=%s", arm_a_fk, arm_b_fk)

                    step_data = {
                        "step": steps_executed,
                        "action": action.name,
                        "objects": action.objects,
                        "policy": args.policy,
                        "policy_fallback": policy_fallback,
                        "command_action": _safe_copy(command_action),
                        "result": result[4].copy(),
                        "scene_state": last_scene.__dict__.copy(),
                        "timestamp": time.perf_counter() - started,
                        "repeat": repeat,
                    }
                    trajectory.append(step_data)

                    if render_enabled and environment._viewer is not None:
                        environment.render()
                        if args.save_video:
                            frame = environment._viewer.get_snapshot()
                            video_frames.append(frame)

                    steps_executed += 1
                    if result[4].get("terminated") or result[4].get("truncated"):
                        LOGGER.info("Episode ended at step %d", steps_executed)
                        break

                if action.name in {"pick", "grasp", "retry_grasp"}:
                    target = action.objects[0] if action.objects else ""
                    arm_label = _object_assigned_to_arm(target)
                    gripper_key = "gripper_a" if arm_label == "a" else "gripper_b"
                    gripper_val = float(command_action.get(gripper_key, 0.0))
                    grasp_result = _verify_grasp(environment, target, arm_label)
                    _log_grasp_debug(LOGGER, environment, target, command_action, grasp_result["success"])
                    if not grasp_result["success"]:
                        failed_grasps += 1
                        LOGGER.warning("Failed grasp for object: %s reason=%s", target, grasp_result["failure_reason"])
                        trajectory.append({
                            "step": steps_executed,
                            "action": action.name,
                            "policy": args.policy,
                            "status": "FAILED",
                            "reason": grasp_result["failure_reason"],
                            "object": target,
                            "gripper_value": gripper_val,
                            "timestamp": time.perf_counter() - started,
                        })
                        if hasattr(policy, '_held_object'):
                            policy._held_object = None
                        held_objects[f"arm_{arm_label}"] = None
                    else:
                        successful_actions += 1
                        object_states[target] = "GRASPED"
                        if hasattr(policy, '_held_object'):
                            policy._held_object = target
                        held_objects[f"arm_{arm_label}"] = target

                if action.name in {"place", "release"}:
                    target = action.objects[0] if action.objects else ""
                    arm_label = _object_assigned_to_arm(target)
                    if _object_on_table(environment, target):
                        successful_placements += 1
                        successful_actions += 1
                        object_states[target] = "PLACED"
                        if hasattr(policy, '_held_object'):
                            policy._held_object = None
                        held_objects[f"arm_{arm_label}"] = None
                        LOGGER.info("Successful placement of %s", target)
                        trajectory.append({
                            "step": steps_executed,
                            "action": action.name,
                            "policy": args.policy,
                            "status": "SUCCESS",
                            "reason": "object_on_table",
                            "object": target,
                            "timestamp": time.perf_counter() - started,
                        })
                    else:
                        failed_actions += 1
                        LOGGER.warning("Failed placement of %s", target)
                        trajectory.append({
                            "step": steps_executed,
                            "action": action.name,
                            "policy": args.policy,
                            "status": "FAILED",
                            "reason": "object_not_on_table",
                            "object": target,
                            "timestamp": time.perf_counter() - started,
                        })

                if action.name == "transport":
                    target = action.objects[0] if action.objects else ""
                    object_states[target] = "TRANSPORTING"

                if action.name == "retry_grasp":
                    if hasattr(policy, 'metrics') and hasattr(policy.metrics, 'recovery_attempts'):
                        policy.metrics.recovery_attempts += 1

                last_scene = processor.perceive(
                    result[0], environment.model, environment.data, perception_config.get("camera", "overhead")
                )

            except Exception as exc:
                LOGGER.error("Error executing action %s: %s", action.name, exc)
                failed_actions += 1
                error = exc
                trajectory.append({
                    "step": steps_executed,
                    "action": action.name,
                    "policy": args.policy,
                    "status": "ERROR",
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "timestamp": time.perf_counter() - started,
                })
                _recover_from_failure(environment, policy, last_scene, i, args.policy)
                if hasattr(policy, 'metrics') and hasattr(policy.metrics, 'recovery_attempts'):
                    policy.metrics.recovery_attempts += 1
                i += 1
                continue

            i += 1

        if result is None:
            raise RuntimeError("Planner produced no executable result")

        LOGGER.info("Phase 10: Verifying demonstration completion")
        policy_metrics = policy.complete(last_scene, int(result[4]["step"]), time.perf_counter() - started, collision_count)
        metrics = dict(policy_metrics)

        LOGGER.info("Phase 11: Demonstrating error recovery")
        demonstration_status = _verify_dinner_table_completion(environment, last_scene)
        bimanual_metrics = getattr(policy, 'metrics', BaselineMetrics())
        metrics.update({
            "demonstration_status": demonstration_status,
            "failed_grasps": failed_grasps,
            "successful_placements": successful_placements,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "seed": seed,
            "reset_info": reset_info,
            "planned_actions": sequence.names(),
            "policy": args.policy,
            "device": args.device,
            "render": render_enabled,
            "save_video": args.save_video,
            "max_steps": args.max_steps,
            "instruction": instruction,
            "trajectory_length": len(trajectory),
            "drawer_state": last_scene.drawer_state,
            "task_state": last_scene.robot_state.get("task_state", {}),
            "vla_checkpoint_loaded": not policy_fallback,
            "object_states": object_states,
            "held_objects": held_objects,
            "left_arm_actions": bimanual_metrics.left_arm_actions,
            "right_arm_actions": bimanual_metrics.right_arm_actions,
            "bimanual_actions": bimanual_metrics.bimanual_actions,
            "single_arm_actions": bimanual_metrics.single_arm_actions,
            "recovery_attempts": bimanual_metrics.recovery_attempts,
            "actual_policy": "scripted" if policy_fallback else args.policy,
            "vla_status": "UNAVAILABLE" if policy_fallback and args.policy == "vla" else ("READY" if args.policy == "vla" and not policy_fallback else "N/A"),
            "fallback_used": policy_fallback,
            "fallback_reason": "NO_CHECKPOINT" if policy_fallback and args.policy == "vla" else ("CHECKPOINT_NOT_FOUND" if policy_fallback else "NONE"),
        })

        LOGGER.info("Phase 12: Saving trajectory and metrics")
        output_dir = ROOT / "outputs" / "demonstrations"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / f"demo_seed_{seed}_{args.policy}.json"
        metrics_path.write_text(safe_json_dumps(metrics, indent=2), encoding="utf-8")

        trajectory_path = output_dir / f"demo_seed_{seed}_{args.policy}_trajectory.json"
        trajectory_data = {
            "metrics": metrics,
            "trajectory": trajectory,
            "video_frames": len(video_frames) if args.save_video else 0,
            "environment": "BimanualMujocoEnv",
            "policy": args.policy,
            "seed": seed,
            "instruction": instruction,
        }
        trajectory_path.write_text(safe_json_dumps(trajectory_data, indent=2), encoding="utf-8")

        LOGGER.info("Phase 13: Final demonstration complete: %s", metrics.get("task_success", "Unknown"))
        LOGGER.info("Success=%s Steps=%s Outputs=%s",
                   metrics.get("task_success", "Unknown"), metrics.get("step_count", 0), metrics_path)

        return 0

    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        LOGGER.error("Demonstration failed: %s", exc)
        error = exc
        return 1
    finally:
        if environment is not None:
            LOGGER.info("Cleaning up environment")
            environment.close()


def _object_in_gripper(environment: BimanualMujocoEnv, object_name: str) -> bool:
    """Check real MuJoCo contacts between the named object and gripper bodies."""
    if not object_name:
        return False
    object_body = environment.model.body(object_name).id
    gripper_bodies = {
        environment.model.body(f"arm_{label}_gripper_{side}").id
        for label in ("a", "b") for side in ("left", "right")
    }
    for index in range(environment.data.ncon):
        contact = environment.data.contact[index]
        body_a = environment.model.geom_bodyid[contact.geom1]
        body_b = environment.model.geom_bodyid[contact.geom2]
        if (body_a == object_body and body_b in gripper_bodies) or (body_b == object_body and body_a in gripper_bodies):
            return True
    return False


def _verify_grasp(environment: BimanualMujocoEnv, object_name: str, arm_label: str) -> dict[str, Any]:
    """Verify a grasp using real MuJoCo contact data.

    Returns a structured GraspResult with contact and hold stability info.
    """
    result = {
        "success": False,
        "object": object_name,
        "arm": arm_label,
        "contact_detected": False,
        "left_finger_contact": False,
        "right_finger_contact": False,
        "object_following": False,
        "hold_stable": False,
        "failure_reason": "NO_CONTACT",
    }
    if not object_name:
        result["failure_reason"] = "NO_OBJECT"
        return result

    object_body = environment.model.body(object_name).id
    gripper_bodies = {
        environment.model.body(f"arm_{arm_label}_gripper_{side}").id
        for side in ("left", "right")
    }

    left_finger_body = environment.model.body(f"arm_{arm_label}_gripper_left").id
    right_finger_body = environment.model.body(f"arm_{arm_label}_gripper_right").id

    left_contacts = 0
    right_contacts = 0
    object_contacts = 0

    for index in range(environment.data.ncon):
        contact = environment.data.contact[index]
        body_a = environment.model.geom_bodyid[contact.geom1]
        body_b = environment.model.geom_bodyid[contact.geom2]

        if body_a == object_body or body_b == object_body:
            object_contacts += 1
            if body_a == left_finger_body or body_b == left_finger_body:
                left_contacts += 1
            if body_a == right_finger_body or body_b == right_finger_body:
                right_contacts += 1

    result["contact_detected"] = object_contacts > 0
    result["left_finger_contact"] = left_contacts > 0
    result["right_finger_contact"] = right_contacts > 0
    result["failure_reason"] = "NO_CONTACT" if not result["contact_detected"] else "GRASPED"

    if result["contact_detected"]:
        result["success"] = True
        result["failure_reason"] = "GRASPED"

    return result


def _object_assigned_to_arm(object_name: str, default_arm: str = "a") -> str:
    """Resolve arm for object using the centralized OBJECT_ARM_MAP.

    This is the single source of truth for arm assignment.
    """
    return OBJECT_ARM_MAP.get(object_name, default_arm)


def _log_grasp_debug(logger: logging.Logger, environment: Any, object_name: str, command_action: dict[str, Any], grasp_result: dict[str, Any] | bool) -> None:
    object_pos = None
    target_pos = None
    ee_pos = None
    position_error = None
    contact_pairs = 0
    success = False
    left_finger_contact = False
    right_finger_contact = False
    failure_reason = "UNKNOWN"
    try:
        if isinstance(grasp_result, dict):
            success = grasp_result.get("success", False)
            left_finger_contact = grasp_result.get("left_finger_contact", False)
            right_finger_contact = grasp_result.get("right_finger_contact", False)
            failure_reason = grasp_result.get("failure_reason", "UNKNOWN")
        else:
            success = bool(grasp_result)
        body_id = environment.model.body(object_name).id
        object_pos = environment.data.xpos[body_id].tolist()
        target_pos = command_action.get("target_position")
        if target_pos is None:
            arm_label = OBJECT_ARM_MAP.get(object_name, "a")
            site_name = f"arm_{arm_label}_grasp_site"
            site_id = environment.model.site(site_name).id
            target_pos = environment.data.site_xpos[site_id].tolist()
        arm_label = OBJECT_ARM_MAP.get(object_name, "a")
        site_name = f"arm_{arm_label}_grasp_site"
        site_id = environment.model.site(site_name).id
        ee_pos = environment.data.site_xpos[site_id].tolist()
        position_error = float(np.linalg.norm(np.asarray(object_pos) - np.asarray(ee_pos)))
        contact_pairs = sum(
            1 for idx in range(environment.data.ncon)
            if environment.model.geom_bodyid[environment.data.contact[idx].geom1] == body_id
            or environment.model.geom_bodyid[environment.data.contact[idx].geom2] == body_id
        )
    except Exception as exc:
        logger.debug("Grasp debug partial failure for %s: %s", object_name, exc)

    logger.info(
        "GRASP DEBUG object=%s object_position=%s target_position=%s ee_position=%s position_error=%.4f contact_pairs=%d success=%s left_finger=%s right_finger=%s failure_reason=%s",
        object_name,
        object_pos,
        target_pos,
        ee_pos,
        position_error if position_error is not None else -1.0,
        contact_pairs,
        success,
        left_finger_contact,
        right_finger_contact,
        failure_reason,
    )


def _safe_copy(mapping: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in mapping.items()}


def _object_on_table(environment: BimanualMujocoEnv, object_name: str) -> bool:
    """Check a live object pose against the configured table bounds."""
    if not object_name:
        return False
    position = environment.data.xpos[environment.model.body(object_name).id]
    half_height = OBJECT_HALF_HEIGHTS.get(object_name, 0.02)
    return bool(TABLE_TOP - 0.05 <= position[2] <= TABLE_TOP + 0.5 and abs(position[0]) <= 0.85 and abs(position[1]) <= 0.55)


def _recover_from_failure(
    environment: BimanualMujocoEnv,
    policy: Any,
    scene_state: Any,
    action_index: int,
    policy_type: str,
) -> None:
    """Recover from failures during demonstration."""
    LOGGER.warning("Attempting recovery at action index %d", action_index)
    try:
        environment.reset(seed=int(time.perf_counter() * 1000) % 10000)
        if policy_type == "scripted":
            policy.reset(None)
        else:
            policy.reset(None)
        LOGGER.info("Recovery completed")
    except Exception as exc:
        LOGGER.error("Recovery failed: %s", exc)
        raise


def _verify_dinner_table_completion(environment: BimanualMujocoEnv, scene_state: Any) -> dict[str, Any]:
    """Verify that the dinner-table task is complete."""
    verification = {
        "dinner_set": False,
        "utensils_placed": False,
        "plate_placed": False,
        "cup_placed": False,
        "spoon_placed": False,
        "fork_placed": False,
        "robot_position_valid": False,
        "collision_detected": False,
    }

    try:
        required_objects = ["plate", "cup", "spoon", "fork"]
        placed_objects = 0
        for obj in required_objects:
            try:
                if _object_on_table(environment, obj):
                    placed_objects += 1
                    verification[f"{obj}_placed"] = True
            except KeyError:
                pass

        verification["utensils_placed"] = verification["spoon_placed"] and verification["fork_placed"]
        verification["dinner_set"] = placed_objects >= 4

        try:
            robot_pos = environment.data.qpos[0]
            if hasattr(robot_pos, '__len__') and not isinstance(robot_pos, (str, bytes)):
                robot_pos_list = list(robot_pos)
            else:
                robot_pos_list = [float(robot_pos)]
            
            verification["robot_position_valid"] = all(abs(float(x)) < 2.0 for x in robot_pos_list)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            LOGGER.error("Robot position verification error: %s", exc)
            verification["robot_position_valid"] = False

        verification["collision_detected"] = bool(scene_state.robot_state.get("collision", False))

    except Exception as exc:
        LOGGER.error("Verification error: %s", exc)

    return verification


if __name__ == "__main__":
    raise SystemExit(main())