import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data, QoSProfile
# Foxy har bare de lange navnene (QoSDurabilityPolicy), nyere versjoner de korte
try:
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
except ImportError:
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
    from rclpy.qos import QoSReliabilityPolicy as ReliabilityPolicy


def euler_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RescueRobot(Node):
    def __init__(self):
        super().__init__('rescue_robot')

        # NB: 'use_sim_time' er allerede deklarert av rclpy selv. Å deklarere den
        # på nytt gir ParameterAlreadyDeclaredException og noden krasjer ved oppstart.
        # Den settes fra launch-fila via parameters=[{'use_sim_time': True}].
        # Målet er et punkt ~0.6 m ut fra markøren på finnerens side
        # (regnes ut av koordinatoren), så vi kan stoppe ganske nær det.
        self.declare_parameter('arrive_dist', 0.7)
        self.arrive_dist = self.get_parameter('arrive_dist').value

        ns = self.get_namespace().strip('/') or 'tb3_2'

        # Posisjon hentes fra odom i stedet for TF. Pythons TransformListener
        # abonnerer på det RELATIVE topicet 'tf', som i namespace tb3_2 blir
        # /tb3_2/tf (tomt) i stedet for /tf. Siden map -> tb3_2/odom er en
        # statisk identitets-transform, er odom-posisjonen lik map-posisjonen.
        self.pose = None

        self.state = 'WAITING'
        self.start_pos = None
        self.goal_pos = None
        self.hit_dist_to_goal = float('inf')
        self.wall_follow_start_ns = 0

        self.regions = {
            'right': 10.0,
            'fright': 10.0,
            'front': 10.0,
            'fleft': 10.0,
            'left': 10.0,
        }

        self.side_target = 0.42
        self.front_safe = 0.55
        self.forward_speed = 0.22
        self.turn_speed = 0.65
        self.m_line_tolerance = 0.15

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.clbk_laser, qos_profile_sensor_data
        )
        self.target_sub = self.create_subscription(
            Point, 'fire_target', self.target_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.clbk_odom, 10
        )

        # Kartet brukes til å sjekke at det ikke er en vegg mellom oss og målet.
        # map_server publiserer /map som transient_local (latched).
        self.map = None
        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.clbk_map, map_qos
        )

        self.arrived_pub = self.create_publisher(Bool, 'rescue_arrived', 10)

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f'Redningsrobot ({ns}) klar med Bug2 - venter ved spawn...')

    def clbk_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, euler_from_quaternion(p.orientation))

    def get_robot_pose(self):
        return self.pose

    def clbk_map(self, msg):
        self.map = msg
        self.get_logger().info(
            f'Kart mottatt ({msg.info.width}x{msg.info.height}, '
            f'{msg.info.resolution:.2f} m/celle)')

    def is_occupied(self, x, y):
        info = self.map.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)
        if col < 0 or row < 0 or col >= info.width or row >= info.height:
            return False
        return self.map.data[row * info.width + col] > 50

    def has_line_of_sight(self, x0, y0, x1, y1, stop_short=0.25):
        """Sann hvis ingen vegg i kartet ligger på linja (x0,y0) -> (x1,y1).
        Vi stopper litt før målet i tilfelle målpunktet havner nær/inni veggen
        pga. unøyaktig markørposisjon."""
        if self.map is None:
            self.get_logger().warn('Har ikke kart - antar fri sikt',
                                   throttle_duration_sec=5.0)
            return True
        length = math.hypot(x1 - x0, y1 - y0)
        check_len = max(0.0, length - stop_short)
        step = self.map.info.resolution / 2.0
        n = int(check_len / step)
        for i in range(1, n + 1):
            t = (i * step) / length
            if self.is_occupied(x0 + t * (x1 - x0), y0 + t * (y1 - y0)):
                return False
        return True

    def target_callback(self, msg):
        if self.goal_pos is not None:
            return

        self.goal_pos = (msg.x, msg.y)
        self.get_logger().info(
            f'Mottok SOS! Mål satt til ({msg.x:.2f}, {msg.y:.2f}). Starter...'
        )

    def _get_slice_min(self, ranges, start_deg, end_deg):
        n = len(ranges)
        if n == 0:
            return 10.0

        start_idx = int(start_deg * n / 360.0)
        end_idx = int(end_deg * n / 360.0)

        if start_idx <= end_idx:
            slice_vals = ranges[start_idx:end_idx]
        else:
            slice_vals = ranges[start_idx:] + ranges[:end_idx]

        valid = [r for r in slice_vals if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        return min(valid) if valid else 10.0

    def clbk_laser(self, msg):
        ranges = list(msg.ranges)
        self.regions = {
            'right':  self._get_slice_min(ranges, 270, 305),
            'fright': self._get_slice_min(ranges, 306, 341),
            'front':  self._get_slice_min(ranges, 342, 17),
            'fleft':  self._get_slice_min(ranges, 18, 53),
            'left':   self._get_slice_min(ranges, 54, 90),
        }

    def distance_to_m_line(self, x, y):
        x1, y1 = self.start_pos
        x2, y2 = self.goal_pos
        line_len = math.hypot(x2 - x1, y2 - y1)
        if line_len < 1e-6:
            return 0.0
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        return numerator / line_len

    def is_between_start_and_goal(self, x, y):
        x1, y1 = self.start_pos
        x2, y2 = self.goal_pos
        dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
        sq_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
        return 0.0 <= dot <= sq_len

    def control_loop(self):
        msg = Twist()

        if self.state == 'ARRIVED':
            self.arrived_pub.publish(Bool(data=True))
            self.cmd_pub.publish(msg)
            return

        if self.goal_pos is None:
            self.cmd_pub.publish(msg)
            return

        pose = self.get_robot_pose()
        if pose is None:
            self.get_logger().warn(
                'Venter fortsatt på odom...',
                throttle_duration_sec=2.0
            )
            self.cmd_pub.publish(msg)
            return
        if self.state == 'WAITING' or self.start_pos is None:
            self.start_pos = (pose[0], pose[1])
            self.state = 'GO_TO_POINT'
            self.get_logger().info(
                f'Posisjon klar! Starter Bug2 fra ({pose[0]:.2f}, {pose[1]:.2f}) '
                f'mot ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})'
            )

        x, y, yaw = pose
        gx, gy = self.goal_pos
        dist_to_goal = math.hypot(gx - x, gy - y)

        # Fremme = nær målet OG ingen vegg mellom oss og målet. Står vi på feil
        # side av veggen fortsetter Bug2 (veggfølging) til vi kommer rundt.
        if dist_to_goal < self.arrive_dist:
            if self.has_line_of_sight(x, y, gx, gy):
                self.state = 'ARRIVED'
                self.get_logger().info('Bug2: Fremme ved brannen! Stopper.')
                self.arrived_pub.publish(Bool(data=True))
                self.cmd_pub.publish(msg)
                return
            self.get_logger().info(
                'Nær målet, men vegg i mellom - fortsetter rundt.',
                throttle_duration_sec=3.0)

        front_blocked = (
            self.regions['front'] < self.front_safe
            or self.regions['fleft'] < 0.38
            or self.regions['fright'] < 0.38
        )

        now = self.get_clock().now().nanoseconds

        if self.state == 'GO_TO_POINT':
            if front_blocked:
                self.state = 'WALL_FOLLOW'
                self.hit_dist_to_goal = dist_to_goal
                self.wall_follow_start_ns = now
                self.get_logger().info(
                    f'Bug2: Hindring møtt ({dist_to_goal:.2f} m fra mål) -> WALL_FOLLOW'
                )
            else:
                target_yaw = math.atan2(gy - y, gx - x)
                heading_err = normalize_angle(target_yaw - yaw)

                if abs(heading_err) > 0.25:
                    msg.linear.x = 0.0
                    msg.angular.z = self.turn_speed if heading_err > 0 else -self.turn_speed
                else:
                    msg.linear.x = self.forward_speed
                    msg.angular.z = max(-self.turn_speed, min(1.2 * heading_err, self.turn_speed))

        elif self.state == 'WALL_FOLLOW':
            time_on_wall = now - self.wall_follow_start_ns
            dist_mline = self.distance_to_m_line(x, y)

            if (time_on_wall > 1_500_000_000
                    and dist_mline < self.m_line_tolerance
                    and dist_to_goal < (self.hit_dist_to_goal - 0.20)
                    and self.is_between_start_and_goal(x, y)):
                self.state = 'GO_TO_POINT'
                self.get_logger().info(
                    f'Bug2: Tilbake på M-linjen ({dist_to_goal:.2f} m fra mål) -> GO_TO_POINT'
                )
            else:
                left = self.regions['left']
                if front_blocked:
                    msg.linear.x = 0.04
                    msg.angular.z = -self.turn_speed
                elif left > 1.5:
                    msg.linear.x = self.forward_speed * 0.85
                    msg.angular.z = self.turn_speed * 0.8
                else:
                    error = self.side_target - left
                    msg.linear.x = self.forward_speed
                    msg.angular.z = max(
                        -self.turn_speed, min(-1.2 * error, self.turn_speed)
                    )

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RescueRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()