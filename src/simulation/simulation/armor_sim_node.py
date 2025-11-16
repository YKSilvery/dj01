import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from aimbot_msgs.msg import Detection, DetectionArray
from geometry_msgs.msg import Point32, Polygon


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ArmorSimulationNode(Node):
    def __init__(self) -> None:
        super().__init__('armor_simulation_node')

        self.publish_rate = float(self.declare_parameter('publish_rate', 60.0).value)
        angular_velocity_deg = float(self.declare_parameter('angular_velocity_deg', 60.0).value)
        self.angular_velocity = math.radians(angular_velocity_deg)
        initial_yaw_deg = float(self.declare_parameter('initial_yaw_deg', 0.0).value)
        self.current_yaw = math.radians(initial_yaw_deg)

        self.pitch_rad = math.radians(float(self.declare_parameter('pitch_deg', -15.0).value))
        self.radius_upper = float(self.declare_parameter('radius_upper_mm', 200.0).value)
        self.radius_lower = float(self.declare_parameter('radius_lower_mm', 200.0).value)
        self.z_upper = float(self.declare_parameter('z_upper_mm', 120.0).value)
        self.z_lower = float(self.declare_parameter('z_lower_mm', -120.0).value)

        self.center_x = float(self.declare_parameter('center_x_mm', 1000.0).value)
        self.center_y = float(self.declare_parameter('center_y_mm', 0.0).value)
        self.center_z = float(self.declare_parameter('center_z_mm', 0.0).value)

        self.camera_heading = math.radians(float(self.declare_parameter('camera_heading_deg', 0.0).value))
        visible_min_deg = float(self.declare_parameter('visible_yaw_min_deg', -70.0).value)
        visible_max_deg = float(self.declare_parameter('visible_yaw_max_deg', 70.0).value)
        self.visible_yaw_min = math.radians(visible_min_deg)
        self.visible_yaw_max = math.radians(visible_max_deg)

        category_value = str(self.declare_parameter('publish_category', 'upper').value).lower()
        if category_value not in ('upper', 'lower'):
            self.get_logger().warning('publish_category must be "upper" or "lower", defaulting to "upper"')
            category_value = 'upper'
        self.publish_category = category_value

        self.class_id = int(self.declare_parameter('class_id', 0).value)
        self.class_name = str(self.declare_parameter('class_name', 'simulated').value)
        self.confidence = float(self.declare_parameter('confidence', 1.0).value)
        self.topic_name = str(self.declare_parameter('topic', 'detections').value)

        self.publisher = self.create_publisher(DetectionArray, self.topic_name, 10)

        self.base_offsets: List[Tuple[str, float]] = [
            ('front', 0.0),
            ('right', -math.pi / 2.0),
            ('back', math.pi),
            ('left', math.pi / 2.0),
        ]

        period = 1.0 / self.publish_rate if self.publish_rate > 1e-6 else 0.02
        self.timer = self.create_timer(period, self._on_timer)
        self.last_update_time = self.get_clock().now()
        self.base_offsets: List[Tuple[str, float]] = [
            ('front', 0.0),
            ('right', -math.pi / 2.0),
            ('back', math.pi),
            ('left', math.pi / 2.0),
        ]

        if self.armor_category == 'outpost':
            # Outpost has 3 armor plates at 120-degree intervals
            self.base_offsets = [
                ('armor1', 0.0),
                ('armor2', 2.0 * math.pi / 3.0),
                ('armor3', 4.0 * math.pi / 3.0),
            ]

        period = 1.0 / self.publish_rate if self.publish_rate > 1e-6 else 0.02
        self.timer = self.create_timer(period, self._on_timer)
        self.last_update_time = self.get_clock().now()

        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params) -> SetParametersResult:
        for param in params:
            if param.name == 'publish_rate':
                self.publish_rate = float(param.value)
                self.timer.cancel()
                period = 1.0 / self.publish_rate if self.publish_rate > 1e-6 else 0.02
                self.timer = self.create_timer(period, self._on_timer)
            elif param.name == 'angular_velocity_deg':
                self.angular_velocity = math.radians(float(param.value))
            elif param.name == 'pitch_deg':
                self.pitch_rad = math.radians(float(param.value))
            elif param.name == 'radius_upper_mm':
                self.radius_upper = float(param.value)
            elif param.name == 'radius_lower_mm':
                self.radius_lower = float(param.value)
            elif param.name == 'z_upper_mm':
                self.z_upper = float(param.value)
            elif param.name == 'z_lower_mm':
                self.z_lower = float(param.value)
            elif param.name == 'center_x_mm':
                self.center_x = float(param.value)
            elif param.name == 'center_y_mm':
                self.center_y = float(param.value)
            elif param.name == 'center_z_mm':
                self.center_z = float(param.value)
            elif param.name == 'camera_heading_deg':
                self.camera_heading = math.radians(float(param.value))
            elif param.name == 'visible_yaw_min_deg':
                self.visible_yaw_min = math.radians(float(param.value))
            elif param.name == 'visible_yaw_max_deg':
                self.visible_yaw_max = math.radians(float(param.value))
            elif param.name == 'publish_category':
                category_value = str(param.value).lower()
                if category_value in ('upper', 'lower'):
                    self.publish_category = category_value
                else:
                    self.get_logger().warning(f'Invalid publish_category: {param.value}, ignoring')
            elif param.name == 'armor_category':
                category_value = str(param.value).lower()
                if category_value in ('regular', 'outpost'):
                    self.armor_category = category_value
                    # Update base_offsets based on category
                    if category_value == 'outpost':
                        self.base_offsets = [
                            ('armor1', 0.0),
                            ('armor2', 2.0 * math.pi / 3.0),
                            ('armor3', 4.0 * math.pi / 3.0),
                        ]
                    else:
                        self.base_offsets = [
                            ('front', 0.0),
                            ('right', -math.pi / 2.0),
                            ('back', math.pi),
                            ('left', math.pi / 2.0),
                        ]
                    # Update class_id if it was auto-set
                    if self.class_id == 5 and category_value == 'regular':
                        self.class_id = 0
                    elif self.class_id == 0 and category_value == 'outpost':
                        self.class_id = 5
                else:
                    self.get_logger().warning(f'Invalid armor_category: {param.value}, ignoring')
            elif param.name == 'class_id':
                self.class_id = int(param.value)
            elif param.name == 'class_name':
                self.class_name = str(param.value)
            elif param.name == 'confidence':
                self.confidence = float(param.value)
            elif param.name == 'topic':
                self.topic_name = str(param.value)
                # Note: Changing topic requires recreating publisher, but for simplicity, log warning
                self.get_logger().warning('Topic change requires node restart to take effect')
        return SetParametersResult(successful=True)

    def _select_visible_armor(self) -> Optional[Tuple[float, Tuple[float, float, float]]]:
        candidates: List[Tuple[float, float, Tuple[float, float, float]]] = []
        for _, offset in self.base_offsets:
            yaw_world = wrap_angle(self.current_yaw + offset)
            relative_yaw = wrap_angle(yaw_world - self.camera_heading)
            if self.visible_yaw_min <= relative_yaw <= self.visible_yaw_max:
                if self.publish_category == 'upper':
                    candidates.append((abs(relative_yaw), yaw_world, self._calc_position(yaw_world, self.radius_upper, self.z_upper, self.center_x, self.center_y, self.center_z)))
                else:
                    candidates.append((abs(relative_yaw), yaw_world, self._calc_position(yaw_world, self.radius_lower, self.z_lower, self.center_x, self.center_y, self.center_z)))

        if not candidates:
            return None

        candidates.sort(key=lambda entry: entry[0])
        _, yaw_world, position = candidates[0]
        return yaw_world, position

    @staticmethod
    def _calc_position(yaw_world: float, radius: float, z_value: float, center_x: float, center_y: float, center_z: float) -> Tuple[float, float, float]:
        x = center_x + radius * math.cos(yaw_world)
        y = center_y + radius * math.sin(yaw_world)
        z = center_z + z_value
        return x, y, z

    def _build_detection(self, stamp, yaw_world: float, position: Tuple[float, float, float]) -> Detection:
        detection = Detection()
        detection.header.stamp = stamp.to_msg()
        detection.header.frame_id = 'world'
        detection.class_id = self.class_id
        detection.class_name = self.class_name
        detection.confidence = float(self.confidence)
        detection.center.x = float(position[0])
        detection.center.y = float(position[1])
        detection.center.z = float(position[2])
        half_side = 80.0
        detection.bbox = Polygon(points=[
            Point32(x=-half_side, y=-half_side, z=0.0),
            Point32(x=half_side, y=-half_side, z=0.0),
            Point32(x=half_side, y=half_side, z=0.0),
            Point32(x=-half_side, y=half_side, z=0.0),
        ])

        detection.rotation_vector[0] = 0.0
        detection.rotation_vector[1] = float(self.pitch_rad)
        detection.rotation_vector[2] = float(yaw_world+ math.pi)

        detection.translation_vector[0] = float(position[0])
        detection.translation_vector[1] = float(position[1])
        detection.translation_vector[2] = float(position[2])
        return detection


def main() -> None:
    rclpy.init()
    node = ArmorSimulationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
