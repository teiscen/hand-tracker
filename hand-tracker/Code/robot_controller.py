import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from parser_model import calculate_joint_angles


class _LM:
    """Lightweight landmark wrapper so parser_model can use .x/.y/.z."""

    __slots__ = ("x", "y", "z")

    def __init__(self, d):
        self.x = d["x"]
        self.y = d["y"]
        self.z = d["z"]


class RobotHandROS2Controller(Node):
    def __init__(self):
        super().__init__("robot_hand_controller")
        # Must match joint names in BionicHand.xacro
        self.joint_names = [
            "thumb_lower_joint",
            "thumb_upper_joint",
            "index_lower_joint",
            "index_upper_joint",
            "middle_lower_joint",
            "middle_upper_joint",
            "ring_lower_joint",
            "ring_upper_joint",
            "pinky_lower_joint",
            "pinky_upper_joint",
        ]
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.sub = self.create_subscription(
            String, "/hand_landmarks", self.on_landmarks, 10
        )

    def on_landmarks(self, msg):
        try:
            hands = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not hands:
            return

        landmarks = [_LM(d) for d in hands[0]]
        if len(landmarks) < 21:
            return

        angles_dict = calculate_joint_angles(landmarks)

        angles = []
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            fa = angles_dict.get(f"{finger}_angles", [0.0, 0.0])
            angles.extend([float(fa[0]), float(fa[1])])

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = self.joint_names
        out.position = angles
        self.pub.publish(out)


def main():
    rclpy.init()
    node = RobotHandROS2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
