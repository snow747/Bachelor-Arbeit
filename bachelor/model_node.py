import rclpy
from rclpy.node import Node
from bachelor.motion_model import wheel_to_twist, integrate_pose

class ModelNode(Node):
    def __init__(self):
        super().__init__('model_node')

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.wheel_radius = 0.048
        self.wheel_separation = 0.248

        self.omega_left = 4.0
        self.omega_right = 4.0

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.update_model)

    def update_model(self):
        now = self.get_clock().now()
        dt = (now-self.last_time).nanoseconds/1e9
        self.last_time = now

        if dt <= 0.0:
            return

        v, omega = wheel_to_twist(self.omega_left, 
                                  self.omega_right, 
                                  self.wheel_radius, 
                                  self.wheel_separation
        )
        self.x, self.y, self.yaw = integrate_pose(self.x,
                                   self.y,
                                   self.yaw,
                                   v,
                                   omega,
                                   dt
        )

        self.get_logger().info(
            f'x={self.x:.3f} m, y={self.y:.3f} m, yaw={self.yaw:.3f} rad'
        )

def main(args=None):
    rclpy.init(args=args)
    node = ModelNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()