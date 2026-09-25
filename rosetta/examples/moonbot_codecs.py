# Copyright 2025 Isaac Blankenau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example custom codecs for the showcase contract (contracts/stone.yaml).

A contract channel can name an inline decoder/encoder as
``"module.path:function"`` — normally in your own package. These are the
worked examples stone.yaml points at, bundled so the showcase contract
load-validates anywhere Rosetta is installed. They double as the signature
reference for custom codecs:

- decoder: ``fn(msg, spec) -> np.ndarray | str`` — reduce the message to a
  flat numeric array (or string); the operator pipeline runs after.
- encoder: ``fn(action_vec, spec, stamp_ns) -> msg`` — scatter an
  already-operator-inverted value slice into a ROS message.
"""

from __future__ import annotations

import numpy as np


def decode_battery(msg, spec):
    """std_msgs/Float64MultiArray -> the pack cells we care about (first 4).

    The entry's ``select`` labels these four values and declares the stream's
    width; a decoder must return exactly ``len(spec.names)`` values (checked
    by decode_value).
    """
    del spec
    return np.asarray(msg.data[:4], dtype=np.float32)


def encode_gripper(action_vec, spec, stamp_ns):
    """One action dim -> std_msgs/Float64 gripper command."""
    from std_msgs.msg import Float64

    del spec, stamp_ns
    msg = Float64()
    msg.data = float(action_vec[0])
    return msg

# ---------- Geometry encoders ----------

def encode_twist(
    names: List[str], action_vec: Sequence[float], clamp: Optional[Tuple[float, float]]
):
    """Twist encoder with sensible defaults when names are absent."""
    if names:
        return _encode_via_dotted_paths(
            "geometry_msgs/msg/Twist", names, action_vec, clamp
        )

    # Default mapping when no names specified
    msg_cls = get_message("geometry_msgs/msg/Twist")
    msg = msg_cls()

    # Apply clamping if specified
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])

    # Map to twist fields (linear.x, linear.y, linear.z, angular.x, angular.y, angular.z)
    if len(arr) >= 1:
        msg.linear.x = float(arr[0])
    if len(arr) >= 2:
        msg.angular.z = float(arr[1])  # Common pattern: linear.x, angular.z
    if len(arr) >= 3:
        msg.linear.y = float(arr[2])
    if len(arr) >= 4:
        msg.linear.z = float(arr[3])
    if len(arr) >= 5:
        msg.angular.x = float(arr[4])
    if len(arr) >= 6:
        msg.angular.y = float(arr[5])

    return msg


def encode_wrench_stamped(
    names: List[str],
    action_vec: Sequence[float],
    clamp: Optional[Tuple[float, float]],
):
    """WrenchStamped encoder.

    Naming conventions supported:
    - "wrench.force.x", "wrench.force.y", "wrench.force.z"
    - "wrench.torque.x", "wrench.torque.y", "wrench.torque.z"

    If no names are provided:
    - Default mapping:
        [0:3] -> force (x, y, z)
        [3:6] -> torque (x, y, z)

    Note:
    - header is left default (timestamp/frame can be filled upstream if needed)
    """
    msg_cls = get_message("geometry_msgs/msg/WrenchStamped")
    msg = msg_cls()

    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])

    # If names provided → use dotted-path fallback
    if names:
        return _encode_via_dotted_paths(
            "geometry_msgs/msg/WrenchStamped", names, arr, clamp
        )

    # Default mapping
    if len(arr) >= 1:
        msg.wrench.force.x = float(arr[0])
    if len(arr) >= 2:
        msg.wrench.force.y = float(arr[1])
    if len(arr) >= 3:
        msg.wrench.force.z = float(arr[2])
    if len(arr) >= 4:
        msg.wrench.torque.x = float(arr[3])
    if len(arr) >= 5:
        msg.wrench.torque.y = float(arr[4])
    if len(arr) >= 6:
        msg.wrench.torque.z = float(arr[5])

    return msg


def encode_transform(
    names: List[str],
    action_vec: Sequence[float],
    clamp: Optional[Tuple[float, float]],
):
    """Transform encoder.

    Naming conventions supported:
    - "translation.x", "translation.y", "translation.z"
    - "rotation.x", "rotation.y", "rotation.z", "rotation.w"

    If no names are provided:
    - Default mapping:
        [0:3] -> translation (x, y, z)
        [3:7] -> rotation quaternion (x, y, z, w)

    Notes:
    - If rotation is partially specified, remaining values default to 0.0
    - If w is not provided, it will default to 1.0 (identity quaternion)
    """
    if names:
        return _encode_via_dotted_paths(
            "geometry_msgs/msg/Transform", names, action_vec, clamp
        )

    msg_cls = get_message("geometry_msgs/msg/Transform")
    msg = msg_cls()

    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])

    # Translation
    if len(arr) >= 1:
        msg.translation.x = float(arr[0])
    if len(arr) >= 2:
        msg.translation.y = float(arr[1])
    if len(arr) >= 3:
        msg.translation.z = float(arr[2])

    # Rotation (quaternion)
    if len(arr) >= 4:
        msg.rotation.x = float(arr[3])
    if len(arr) >= 5:
        msg.rotation.y = float(arr[4])
    if len(arr) >= 6:
        msg.rotation.z = float(arr[5])
    if len(arr) >= 7:
        msg.rotation.w = float(arr[6])
    else:
        # Default to identity quaternion if w not provided
        msg.rotation.w = 1.0

    return msg


def decode_multidof_via_names(msg, names: List[str]) -> np.ndarray:
    """Special decoder for MultiDOFCommand messages with values. and values_dot. prefixes."""
    out: List[float] = []
    
    for name in names:
        try:
            if name.startswith("values_dot."):
                # Extract DOF name from "values_dot.dof_name"
                dof_name = name[11:]  # Remove "values_dot." prefix
                if dof_name in msg.dof_names:
                    idx = msg.dof_names.index(dof_name)
                    if idx < len(msg.values_dot):
                        out.append(float(msg.values_dot[idx]))
                    else:
                        out.append(0.0)
                else:
                    out.append(0.0)
            elif name.startswith("values."):
                # Extract DOF name from "values.dof_name"
                dof_name = name[7:]  # Remove "values." prefix
                if dof_name in msg.dof_names:
                    idx = msg.dof_names.index(dof_name)
                    if idx < len(msg.values):
                        out.append(float(msg.values[idx]))
                    else:
                        out.append(0.0)
                else:
                    out.append(0.0)
            else:
                # Default to values field
                if name in msg.dof_names:
                    idx = msg.dof_names.index(name)
                    if idx < len(msg.values):
                        out.append(float(msg.values[idx]))
                    else:
                        out.append(0.0)
                else:
                    out.append(0.0)
        except Exception:
            out.append(float("nan"))
    
    return np.asarray(out, dtype=np.float32)

def decode_via_names(msg, names: List[str]) -> Optional[np.ndarray]:
    """Fallback: sample scalar fields using dotted selectors into a float32 vector."""
    if not names:
        return None
    
    # Special handling for MultiDOFCommand messages
    if hasattr(msg, 'dof_names') and hasattr(msg, 'values') and hasattr(msg, 'values_dot'):
        return decode_multidof_via_names(msg, names)
    
    out: List[float] = []
    for name in names:
        try:
            out.append(float(dot_get(msg, name)))
        except Exception as e:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float32)

def decode_wrench_stamped(msg, spec):
    """WrenchStamped decoder: try dotted names first, then use default behavior.

    Naming conventions supported (via spec.names):
    - "wrench.force.x", "wrench.force.y", "wrench.force.z"
    - "wrench.torque.x", "wrench.torque.y", "wrench.torque.z"

    Default behavior (no names):
    - Returns [fx, fy, fz, tx, ty, tz]
    """
    # Use explicit mapping if provided
    # if spec.names:
    #     return decode_via_names(msg, spec.names)

    # Default extraction
    values = []
    if hasattr(msg, "wrench"):
        w = msg.wrench

        if hasattr(w, "force"):
            f = w.force
            values.extend([
                float(getattr(f, "x", 0.0)),
                float(getattr(f, "y", 0.0)),
                float(getattr(f, "z", 0.0)),
            ])
        else:
            values.extend([0.0, 0.0, 0.0])

        if hasattr(w, "torque"):
            t = w.torque
            values.extend([
                float(getattr(t, "x", 0.0)),
                float(getattr(t, "y", 0.0)),
                float(getattr(t, "z", 0.0)),
            ])
        else:
            values.extend([0.0, 0.0, 0.0])
    else:
        values = [0.0] * 6

    return np.asarray(values, dtype=np.float32)


def decode_transform(msg, spec):
    """Transform decoder: try dotted names first, then use default behavior.

    Naming conventions supported (via spec.names):
    - "translation.x", "translation.y", "translation.z"
    - "rotation.x", "rotation.y", "rotation.z", "rotation.w"

    Default behavior (no names):
    - Returns [tx, ty, tz, qx, qy, qz, qw]
    """
    # Use explicit mapping if provided
    # if spec.names:
    #     return decode_via_names(msg, spec.names)

    values = []

    # Translation
    if hasattr(msg, "translation"):
        t = msg.translation
        values.extend([
            float(getattr(t, "x", 0.0)),
            float(getattr(t, "y", 0.0)),
            float(getattr(t, "z", 0.0)),
        ])
    else:
        values.extend([0.0, 0.0, 0.0])

    # Rotation (quaternion)
    if hasattr(msg, "rotation"):
        r = msg.rotation
        values.extend([
            float(getattr(r, "x", 0.0)),
            float(getattr(r, "y", 0.0)),
            float(getattr(r, "z", 0.0)),
            float(getattr(r, "w", 1.0)),  # default to identity
        ])
    else:
        values.extend([0.0, 0.0, 0.0, 1.0])

    return np.asarray(values, dtype=np.float32)