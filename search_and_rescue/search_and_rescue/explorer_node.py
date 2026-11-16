#!/usr/bin/env python3

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped, Point, PointStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory

try: 
    from .prm_generator import PRMGenerator
except ImportError:
    from prm_generator import PRMGenerator

#Robotene venter på kart, odom og laser. sender startposisjonen sin og 
# venter på den andre roboten.
#Begge lager den samme PRM-planen og tar hver sin del
#kjører ruta punkt for punkt og ved viewpoints snur den og ser rundt seg.


PACKAGE = (__package__ or 'search_and_rescue').split('.')[0]


def get_yaw_from_quaternion(q):
    #regner robotens retning til yaw
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    #snur alltid den korteste veien
    return math.atan2(math.sin(a), math.cos(a))


LATCHED = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


class ExplorerNode(Node):
    ROBOT_HALF_WIDTH = 0.20
    STOP_DIST = 0.32
    SLOW_DIST = 0.50
    TOO_CLOSE = 0.25
    SIDE_PUSH_DIST = 0.28
    MAX_SPEED = 0.26
    SLOW_SPEED = 0.14
    WAYPOINT_TOL = 0.15
    WAYPOINT_TIMEOUT = 30.0
    STUCK_TIME = 4.0
    BACKUP_TIME = 1.2
    TURN_TIMEOUT = 3.0
    MAX_REPLANS = 3
    TURN_IN_PLACE_ANGLE = 0.4
    PASS_TOL = 0.15
    ROBOT_AVOID_RADIUS = 0.5
    MAX_DEFERS = 3
    DEFER_WAIT = 8.0
    LOOK_DWELL = 0.8
    LOOK_SKIP_SEEN = 0.8
    LOOK_MAX_TIME = 20.0

    def __init__(self):
        # Leser parametre, lager publishers/subscribers, setter startverdiene
        # og starten timerne
        super().__init__('explorer_node')

        self.declare_parameter('namespace', 'tb3_0')
        self.declare_parameter('map_name', '')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('wait_for_peers', 60.0)
        self.declare_parameter('num_robots', 2)
        self.declare_parameter('all_robots', ['tb3_0', 'tb3_1', 'tb3_2'])
        self.declare_parameter('role', 'explorer')

        self.ns = self.get_parameter('namespace').value
        self.role = self.get_parameter('role').value
        self.map_name = self.get_parameter('map_name').value
        self.wait_for_peers = float(self.get_parameter('wait_for_peers').value)
        self.num_robots = int(self.get_parameter('num_robots').value)
        self.map_msg = None
        self.starts = {}
        self.planned_names = None
        self.resplit_pending = False

        prefix = f'/{self.ns}' if self.ns else ''
        self.cmd_vel_pub = self.create_publisher(Twist, f'{prefix}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{prefix}/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, f'{prefix}/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value,
                                 self.map_callback, LATCHED)

        self.start_pub = self.create_publisher(PointStamped, '/explorer/starts', LATCHED)
        self.create_subscription(PointStamped, '/explorer/starts', self.start_callback, LATCHED)

        self.wait_at_fire = False
        self.yielding = False
        self.create_subscription(Bool, f'{prefix}/wait_at_fire', self.wait_callback, 10)
        self.create_subscription(Bool, f'{prefix}/yield', self.yield_callback, 10)
        self.fire_marker = None
        self.fire_goal = None
        self.fire_state = None
        self.fire_attempts = 0
        self.create_subscription(Point, f'{prefix}/fire_approach', self.fire_approach_callback, 10)
        self.at_fire_pub = self.create_publisher(Bool, f'{prefix}/at_fire', 10)
        self.peer_poses = {}
        for other in self.get_parameter('all_robots').value:
            if other != self.ns:
                self.create_subscription(Odometry, f'/{other}/odom',
                                         lambda m, ns=other: self.peer_odom(ns, m), 10)

        self.prm_pub = self.create_publisher(MarkerArray, f'{prefix}/prm_visualization', LATCHED)
        self.path_pub = self.create_publisher(Path, f'{prefix}/exploration_path', LATCHED)
        self.walls_pub = self.create_publisher(MarkerArray, f'{prefix}/seen_walls', LATCHED)

        self.x = self.y = self.yaw = 0.0
        self.have_odom = False

        self.corridor_min = math.inf
        self.front_half_min = math.inf
        self.left_min = self.right_min = self.rear_min = math.inf
        self.have_scan = False

        self.state = 'waiting'
        self.state_start = None
        self.turn_dir = 1.0
        self.replans = 0
        self.replan_target = None
        self.looks = {}
        self.look_queue = []
        self.look_dwell_start = None
        self.looked = set()
        self.deferred = []
        self.defer_count = {}
        self.defer_wait_until = 0.0
        self.progress_pose = None
        self.progress_time = 0.0

        self.prm = None
        self.waypoints = []
        self.wp_index = 0
        self.wp_start_time = 0.0

        self.create_timer(0.1, self.control_loop)
        self.create_timer(2.0, self.publish_viz)
        self.create_timer(0.25, self.update_seen)
        self.get_logger().info(f'[{self.ns}] waiting for map, odometry and laser...')

    def now(self):
        # sim tid i sekunder
        return self.get_clock().now().nanoseconds * 1e-9

    def reset_progress(self):
        # husker pos og tiden nå for stuck sjekken
        self.progress_pose = (self.x, self.y, self.yaw)
        self.progress_time = self.now()

    def check_progress(self):
        # stuck sjekk hvis roboten ikke har flyttet eller snudd seg
        px, py, pyaw = self.progress_pose
        if math.hypot(self.x - px, self.y - py) > 0.05 or abs(wrap_angle(self.yaw - pyaw)) > 0.25:
            self.reset_progress()
        return self.now() - self.progress_time < self.STUCK_TIME

    def next_waypoint(self, reason):
        # Går videre til neste punkt
        self.get_logger().info(f'Waypoint {self.wp_index + 1}/{len(self.waypoints)} {reason}')
        target = tuple(self.waypoints[self.wp_index])
        if self.replan_target is None or target == self.replan_target:
            self.replans, self.replan_target = 0, None
        self.wp_index += 1
        self.wp_start_time = self.now()
        self.reset_progress()
        if self.fire_state == 'approach' and target == self.fire_goal and 'big fire' not in reason:
            self.fire_goal_missed()

    def map_callback(self, msg):
        #lagrer mappet
        if self.map_msg is None:
            self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height} '
                                   f'@ {msg.info.resolution} m')
        self.map_msg = msg

    def start_callback(self, msg):
        # tar startposisjonen til robotene
        ns = msg.header.frame_id
        if ns not in self.starts:
            self.starts[ns] = (msg.point.x, msg.point.y)
            if ns != self.ns:
                self.get_logger().info(f'Peer {ns} starts at ({msg.point.x:.2f}, {msg.point.y:.2f})')
            if self.planned_names is not None and ns not in self.planned_names:
                self.get_logger().warn(f'{ns} joined after we planned - re-splitting the building')
                self.resplit_pending = True

    def peer_odom(self, ns, msg):
        # lagrer posisjonen til de andre robotene
        p = msg.pose.pose.position
        self.peer_poses[ns] = (p.x, p.y)

    FIRE_STANDOFF = 1.0 
    FIRE_OK_DIST = 1.8 

    def fire_approach_callback(self, msg):
        #lagrer markøren til big fire og kaller plan_fire_approach
        if self.fire_state is not None or self.prm is None:
            return
        self.fire_marker = (msg.x, msg.y)
        self.get_logger().info(f'Big fire at ({msg.x:.2f}, {msg.y:.2f}) - driving in close')
        self.plan_fire_approach()

    def plan_fire_approach(self):
        #finner et ledig punkt ved brannen og legger ruten dit 
        mx, my = self.fire_marker
        d_now = math.hypot(self.x - mx, self.y - my)
        if d_now <= self.FIRE_STANDOFF + 0.2:
            self.arrive_at_fire()
            return
        ux, uy = (self.x - mx) / d_now, (self.y - my) / d_now
        goal = None
        for d in (self.FIRE_STANDOFF, 1.2, 0.8, 1.4):
            p = (mx + d * ux, my + d * uy)
            if self.prm.is_free_world(*p, inflated=True):
                goal = p
                break
        if goal is None:
            n = self.prm.nearest_node(mx + self.FIRE_STANDOFF * ux, my + self.FIRE_STANDOFF * uy)
            goal = self.prm._pixel_to_world(*self.prm.nodes[n])
        path = self.prm.plan_path_world((self.x, self.y), goal) or [goal]
        path[-1] = goal
        self.fire_goal = tuple(goal)
        self.fire_state = 'approach'
        self.fire_attempts += 1
        self.look_queue = []
        self.state = 'follow'
        self.replans, self.replan_target = 0, None
        self.waypoints[self.wp_index:self.wp_index] = path
        self.wp_start_time = self.now()
        self.reset_progress()
        self.publish_path_viz()

    def arrive_at_fire(self):
        #gir beskjed til coordinator om ankomst
        self.fire_state = 'at_fire'
        d = math.hypot(self.x - self.fire_marker[0], self.y - self.fire_marker[1])
        self.get_logger().info(f'At the big fire ({d:.2f} m away) - waiting for the rescue robot')
        self.at_fire_pub.publish(Bool(data=True))

    def fire_goal_missed(self):
        #planlegger rute igjen hvis den ikke blir godkjent
        mx, my = self.fire_marker
        if math.hypot(self.x - mx, self.y - my) <= self.FIRE_OK_DIST or self.fire_attempts >= 3:
            self.arrive_at_fire()
        else:
            self.fire_state = None
            self.plan_fire_approach()

    def wait_callback(self, msg):
        #coordinator sier om vi skal vente eller kan fortsette
        if msg.data != self.wait_at_fire:
            self.get_logger().info('Waiting at the big fire' if msg.data else 'Released from the big fire')
        self.wait_at_fire = msg.data

    def yield_callback(self, msg):
        #coordinator sier om vi skal vike for den andre roboten
        if msg.data != self.yielding:
            self.get_logger().info('Giving way to the other robot' if msg.data else 'Continuing')
        self.yielding = msg.data

    def scan_callback(self, msg):
        #sjekker avstander fra lidar
        r = np.asarray(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        a = np.arctan2(np.sin(a), np.cos(a))
        ok = np.isfinite(r) & (r > msg.range_min)
        r, a = r[ok], a[ok]
        x, y = r * np.cos(a), r * np.sin(a)

        def mn(mask):
            #korteste avstand blant mengden
            return float(r[mask].min()) if np.any(mask) else math.inf

        in_corridor = (x > 0.0) & (np.abs(y) < self.ROBOT_HALF_WIDTH)
        self.corridor_min = float(x[in_corridor].min()) if np.any(in_corridor) else math.inf
        self.front_half_min = mn(np.abs(a) < math.pi / 2)
        self.left_min = mn((a > math.radians(20)) & (a < math.radians(110)))
        self.right_min = mn((a < -math.radians(20)) & (a > -math.radians(110)))
        self.rear_min = mn(np.abs(a) > math.radians(135))
        self.have_scan = True

    def odom_callback(self, msg):
        #leser odometry: x, y og yaw
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = get_yaw_from_quaternion(msg.pose.pose.orientation)
        self.have_odom = True

    def startup(self):
        #venter på odom, laser og kart, sender startpos og venter på andre
        #roboten før den planlegger rute
        if not (self.have_odom and self.have_scan):
            return
        if self.map_msg is None and not self.map_name:
            self.get_logger().info('Waiting for /map (is the map server running?)',
                                   throttle_duration_sec=3.0)
            return
        if self.role == 'rescue':
            self.start_rescue_role()
            return
        if self.state_start is None:
            self.state_start = self.now()
            msg = PointStamped()
            msg.header.frame_id = self.ns
            msg.point.x, msg.point.y = self.x, self.y
            self.start_pub.publish(msg)
            self.starts[self.ns] = (self.x, self.y)
            self.get_logger().info(f'[{self.ns}] start pose ({self.x:.2f}, {self.y:.2f}); '
                                   f'waiting up to {self.wait_for_peers:.0f}s for other robots')
        waited = self.now() - self.state_start
        if len(self.starts) < self.num_robots and waited < self.wait_for_peers:
            self.get_logger().info(f'Waiting for the other explorer ({len(self.starts)}/'
                                   f'{self.num_robots} ready, {waited:.0f}s)',
                                   throttle_duration_sec=5.0)
            return
        if len(self.starts) < self.num_robots:
            self.get_logger().warn(f'No other explorer after {self.wait_for_peers:.0f}s - '
                                   f'planning the WHOLE building alone')
        self.build_plan()

    def resplit(self):
        #deler kartet på nytt hvis en robot kommer sent
        self.resplit_pending = False
        if self.fire_state is not None or self.state in ('waiting', 'done'):
            return
        old_seen = self.prm.seen.copy() if hasattr(self.prm, 'seen') else None
        self.deferred, self.look_queue = [], []
        self.replans, self.replan_target = 0, None
        self.build_plan()
        if old_seen is not None and hasattr(self.prm, 'seen') and len(self.prm.seen) == len(old_seen):
            self.prm.seen |= old_seen
        if self.state != 'follow' or not self.waypoints:
            return
        
        k = next((i for i, w in enumerate(self.waypoints)
                  if tuple(w) in self.looks and tuple(w) not in self.looked), None)
        if k is None:
            self.waypoints, self.wp_index = [], 0
            return
        leg = self.prm.plan_path_world((self.x, self.y), self.waypoints[k]) or [self.waypoints[k]]
        leg[-1] = tuple(self.waypoints[k])
        self.waypoints = leg + list(self.waypoints[k + 1:])
        self.wp_index = 0
        self.publish_path_viz()

    def build_plan(self):
        #Lager PRM-roadmappet og regner ut robotens rute og viewpoints
        names = sorted(self.starts)
        self.planned_names = set(names)
        robot_starts = [self.starts[n] for n in names]
        robot_index = names.index(self.ns)
        try:
            if self.map_name:
                share = get_package_share_directory(PACKAGE)
                self.prm = PRMGenerator.from_yaml(os.path.join(share, 'maps', f'{self.map_name}.yaml'))
                src = self.map_name
            else:
                self.prm = PRMGenerator.from_occupancy_grid(self.map_msg)
                src = '/map'
            if not self.prm.is_free_world(self.x, self.y):
                self.get_logger().error(
                    f'Robot pose ({self.x:.2f}, {self.y:.2f}) is inside a wall on the map! '
                    f'Odometry and map frames do not match.')
            self.prm.generate_roadmap(robot_starts)
            rescue = self.peer_poses.get('tb3_2')
            if rescue is not None:
                removed = self.prm.add_obstacle(*rescue, radius=0.25)
                self.get_logger().info(f'Rescue robot at ({rescue[0]:.2f}, {rescue[1]:.2f}) '
                                       f'added as obstacle: removed {removed} roadmap edges')
            self.waypoints, self.looks = self.prm.get_exploration_plan(robot_starts, robot_index)
        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            self.state = 'done'
            return
        L = sum(math.dist(a, b) for a, b in zip(self.waypoints, self.waypoints[1:]))
        self.get_logger().info(
            f'[{self.ns}] map {src}: {self.prm.region_area:.0f} m2 reachable, '
            f'{len(self.prm.nodes)} PRM nodes. Robots {names}, I am #{robot_index}. '
            f'Route: {len(self.waypoints)} waypoints, {L:.1f} m, '
            f'{len(self.looks)} viewpoints covering {self.prm.coverable} wall cells')
        self.publish_prm_viz()
        self.publish_path_viz()
        self.state = 'follow'
        self.wp_start_time = self.now()
        self.reset_progress()

    def start_recovery(self, reason, obstacle_ahead=False):
        #hvis noe er i veien legges det inn i roadmappet og den rygger eller snur
        self.get_logger().warn(f'{reason} -> recovering', throttle_duration_sec=1.0)
        if obstacle_ahead:
            d = self.corridor_min + 0.15
            ox, oy = self.x + d * math.cos(self.yaw), self.y + d * math.sin(self.yaw)
            near_peer = any(math.hypot(ox - px, oy - py) < 0.6 for px, py in self.peer_poses.values())
            removed = 0 if near_peer else self.prm.add_obstacle(ox, oy, radius=0.25)
            if removed:
                self.get_logger().info(f'Unmapped obstacle at ({ox:.2f}, {oy:.2f}): '
                                       f'removed {removed} roadmap edges')
        self.turn_dir = 1.0 if self.left_min > self.right_min else -1.0
        self.state = 'backup' if self.front_half_min < self.TOO_CLOSE else 'turn'
        self.state_start = self.now()

    def finish_recovery(self):
        #planlegger ny vei rundt andre roboter eller hopp over punktet
        self.state = 'follow'
        self.reset_progress()
        self.wp_start_time = self.now()
        if self.replan_target is None:
            self.replan_target = tuple(self.waypoints[self.wp_index])
        end = self.wp_index
        while end < len(self.waypoints) - 1 and tuple(self.waypoints[end]) != self.replan_target:
            end += 1
        self.replans += 1
        if self.replans > self.MAX_REPLANS:
            self.wp_index = end
            self.next_waypoint('skipped (blocked too often)')
            return
        avoid = [(px, py, self.ROBOT_AVOID_RADIUS) for px, py in self.peer_poses.values()
                 if math.hypot(px - self.x, py - self.y) < 4.0]
        tx, ty = self.replan_target
        if any(math.hypot(tx - ax, ty - ay) < self.ROBOT_AVOID_RADIUS for ax, ay, _ in avoid):
            self.wp_index = end
            self.next_waypoint('skipped (another robot is standing on it)')
            return
        new_leg = self.prm.plan_path_world((self.x, self.y), self.replan_target, avoid=avoid)
        if avoid and new_leg:
            self.get_logger().info(f'Replanned around {len(avoid)} robot(s): {len(new_leg)} waypoints')
        elif avoid and self.fire_state != 'approach' and self.role != 'rescue' \
                and self.defer_count.get(self.replan_target, 0) < self.MAX_DEFERS:
            target = self.replan_target
            self.defer_count[target] = self.defer_count.get(target, 0) + 1
            del self.waypoints[self.wp_index:end + 1]
            self.deferred.append(target)
            self.replans, self.replan_target = 0, None
            self.get_logger().warn('Way blocked by another robot - postponing this viewpoint')
            self.publish_path_viz()
            return
        elif avoid and self.role == 'rescue':
            self.waypoints, self.wp_index = [], 0 
            self.rescue_retry_at = self.now() + 1.0
            return
        elif avoid:
            self.get_logger().warn('No route around the other robot(s) - trying the direct route')
            new_leg = self.prm.plan_path_world((self.x, self.y), self.replan_target)
        if new_leg:
            new_leg[-1] = self.replan_target
            self.waypoints[self.wp_index:end + 1] = new_leg
            self.publish_path_viz()
            if not avoid:
                self.get_logger().info(f'Replanned: {len(new_leg)} waypoints to current target')
        else:
            self.wp_index = end
            self.next_waypoint('skipped (no route)')

    def run_recovery(self, cmd):
        #rygger og snir til veien foran er fri
        elapsed = self.now() - self.state_start
        if self.state == 'backup':
            if elapsed < self.BACKUP_TIME and self.rear_min > 0.20:
                cmd.linear.x = -0.10
            else:
                self.state, self.state_start = 'turn', self.now()
        if self.state == 'turn':
            if self.corridor_min > self.SLOW_DIST or elapsed > self.TURN_TIMEOUT:
                self.finish_recovery()
            else:
                cmd.angular.z = 0.8 * self.turn_dir

    def update_seen(self):
        #markerer veggene kameraet ser som ssett
        if self.prm is not None and self.have_odom and hasattr(self.prm, 'seen'):
            self.prm.camera_view(self.x, self.y, self.yaw)
        if self.fire_state == 'at_fire' and self.wait_at_fire:
            self.at_fire_pub.publish(Bool(data=True)) 

    def run_look(self, cmd):
        # Ved et viewpoint snur den mot alle vegger 
        while self.look_queue and self.look_dwell_start is None:
            _, ids = self.look_queue[0]
            if self.prm.seen[ids].mean() >= self.LOOK_SKIP_SEEN:
                self.look_queue.pop(0) 
            else:
                break
        if not self.look_queue or self.now() - self.state_start > self.LOOK_MAX_TIME:
            self.state = 'follow'
            self.look_queue = []
            self.next_waypoint('reached (viewpoint)')
            return
        heading, ids = self.look_queue[0]
        err = wrap_angle(heading - self.yaw)
        if self.look_dwell_start is None and abs(err) > 0.08:
            w = 1.5 * err
            cmd.angular.z = math.copysign(min(max(abs(w), 0.2), 0.8), w)
            return
        if self.look_dwell_start is None:
            self.look_dwell_start = self.now()
        if self.now() - self.look_dwell_start >= self.LOOK_DWELL:
            self.prm.seen[ids] = True
            self.look_queue.pop(0)
            self.look_dwell_start = None

    def resume_deferred(self):
        #prøver viewpoints som evt ble utsatt etter ruten er ferdig
        if self.now() < self.defer_wait_until:
            return False
        target = self.deferred.pop(0)
        avoid = [(px, py, self.ROBOT_AVOID_RADIUS) for px, py in self.peer_poses.values()
                 if math.hypot(px - self.x, py - self.y) < 4.0]
        leg = self.prm.plan_path_world((self.x, self.y), target, avoid=avoid)
        if not leg and self.defer_count.get(target, 0) < self.MAX_DEFERS:
            self.defer_count[target] = self.defer_count.get(target, 0) + 1
            self.deferred.append(target)
            self.defer_wait_until = self.now() + self.DEFER_WAIT
            self.get_logger().info(f'{len(self.deferred)} postponed viewpoint(s) still blocked - waiting',
                                   throttle_duration_sec=5.0)
            return False
        leg = leg or self.prm.plan_path_world((self.x, self.y), target) or [target]
        leg[-1] = target
        self.get_logger().info(f'Returning to a postponed viewpoint ({len(self.deferred)} more left)')
        self.waypoints.extend(leg)
        self.wp_start_time = self.now()
        self.reset_progress()
        self.publish_path_viz()
        return True

    RESCUE_ARRIVE_DIST = 0.9
    RESCUE_RETRY = 3.0
    RESCUE_MARKER_DIST = 1.6

    def start_rescue_role(self):
        #lager roadmappet til rescue roboten og venter til stor brann er funnet
        try:
            self.prm = PRMGenerator.from_occupancy_grid(self.map_msg)
            self.prm.generate_roadmap([(self.x, self.y)])
        except Exception as e:
            self.get_logger().error(f'Rescue planning failed: {e}')
            self.state = 'done'
            return
        self.rescue_target = None
        self.rescue_marker = None
        self.rescue_arrived = False
        self.rescue_retry_at = 0.0
        self.waypoints, self.wp_index = [], 0
        self.rescue_arrived_pub = self.create_publisher(Bool, f'/{self.ns}/rescue_arrived', 10)
        self.create_subscription(Point, f'/{self.ns}/fire_target', self.rescue_target_callback, 10)
        self.create_subscription(Point, f'/{self.ns}/fire_marker', self.rescue_marker_callback, 10)
        self.state = 'rescue_idle'
        self.get_logger().info(f'[{self.ns}] rescue robot ready ({len(self.prm.nodes)} PRM nodes), '
                               f'waiting for the big fire')

    def rescue_target_callback(self, msg):
        #tar imot target fra coordinator
        if self.rescue_target is not None:
            return
        self.rescue_target = (msg.x, msg.y)
        self.rescue_called_at = self.now()
        self.get_logger().info('Big fire! Heading out')

    def choose_parking_spot(self, avoid):
        #finner en parkeringsplass og korteste ruten dit
        mx, my = self.rescue_marker
        prm = self.prm
        cost, _ = prm.dijkstra(prm.nearest_node(self.x, self.y))
        cands = []
        for n, c in cost.items():
            wx, wy = prm._pixel_to_world(*prm.nodes[n])
            d = math.hypot(wx - mx, wy - my)
            if not (0.5 <= d <= 1.45):
                continue
            if any(math.hypot(wx - px, wy - py) < self.ROBOT_AVOID_RADIUS + 0.1
                   for px, py in self.peer_poses.values()):
                continue
            ex, ey = mx + 0.15 * (wx - mx) / d, my + 0.15 * (wy - my) / d
            if prm._check_collision(prm.nodes[n], prm._world_to_pixel(ex, ey), prm.raw_free):
                continue
            cands.append((c, (wx, wy)))
        for _, spot in sorted(cands)[:8]:
            route = prm.plan_path_world((self.x, self.y), spot, avoid=avoid)
            if route:
                return spot, route
        return None, None

    def plan_rescue_route(self):
        #lager rute til parkeringsplassen eller coordinator sitt target
        avoid = [(px, py, self.ROBOT_AVOID_RADIUS) for px, py in self.peer_poses.values()]
        route = None
        if self.rescue_marker is not None:
            spot, route = self.choose_parking_spot(avoid)
            if route:
                self.get_logger().info(f'Parking spot at the fire: ({spot[0]:.2f}, {spot[1]:.2f}), '
                                       f'{math.dist(spot, self.rescue_marker):.2f} m from the marker')
        if not route:
            route = (self.prm.plan_path_world((self.x, self.y), self.rescue_target, avoid=avoid)
                     or self.prm.plan_path_world((self.x, self.y), self.rescue_target))
        if not route:
            self.get_logger().warn('No route to the fire yet - retrying')
            self.rescue_retry_at = self.now() + self.RESCUE_RETRY
            self.waypoints, self.wp_index = [], 0
            return
        if not self.prm.is_free_world(*route[-1], inflated=True) and len(route) > 1:
            route = route[:-1]
        self.waypoints, self.wp_index = route, 0
        self.replans, self.replan_target = 0, None
        self.state = 'follow'
        self.wp_start_time = self.now()
        self.reset_progress()
        self.publish_path_viz()

    def run_rescue(self, cmd):
        #kjører til brannen og sier ifra når den er fremem
        if self.state == 'rescue_idle' and self.rescue_target is None:
            return True
        if self.rescue_arrived:
            self.rescue_arrived_pub.publish(Bool(data=True))
            return True
        if self.rescue_target is None:
            return True
        d = math.hypot(self.x - self.rescue_target[0], self.y - self.rescue_target[1])
        dm = (math.hypot(self.x - self.rescue_marker[0], self.y - self.rescue_marker[1])
              if self.rescue_marker else math.inf)
        if d < self.RESCUE_ARRIVE_DIST or dm < self.RESCUE_MARKER_DIST:
            self.rescue_arrived = True
            self.get_logger().info(f'Arrived at the big fire ({dm:.2f} m from the marker)')
            self.rescue_arrived_pub.publish(Bool(data=True))
            return True
        if self.wp_index >= len(self.waypoints):
            if self.state in ('backup', 'turn'):
                self.state = 'follow'
            if self.rescue_marker is None and self.now() - self.rescue_called_at < 2.0:
                return True
            if self.now() >= self.rescue_retry_at:
                self.plan_rescue_route()
            return True
        return False

    def rescue_marker_callback(self, msg):
        #lagrer big fire markør
        self.rescue_marker = (msg.x, msg.y)

    def control_loop(self):
        #velger hva roboten skal gjøre (vente, rygge, se rudnt, kjøre)
        cmd = Twist()
        if self.state == 'waiting':
            self.startup()
            self.cmd_vel_pub.publish(cmd)
            return
        if self.role == 'rescue' and self.state != 'done' and self.run_rescue(cmd):
            self.cmd_vel_pub.publish(cmd)
            return
        if self.resplit_pending:
            self.resplit()
        if self.state == 'done' and self.fire_state == 'approach':
            self.state = 'follow'
        if self.state != 'done' and self.wp_index >= len(self.waypoints) and self.deferred:
            if not self.resume_deferred():
                self.cmd_vel_pub.publish(cmd)
                return
        if self.state == 'done' or self.wp_index >= len(self.waypoints):
            if self.state != 'done':
                self.get_logger().info('Finished all waypoints!')
                self.state = 'done'
            self.cmd_vel_pub.publish(cmd)
            return
        if (self.wait_at_fire or self.yielding) and self.fire_state != 'approach':
            self.wp_start_time = self.now()
            self.reset_progress()
            if self.state == 'look': 
                self.state_start += 0.1
                if self.look_dwell_start is not None:
                    self.look_dwell_start += 0.1
            self.cmd_vel_pub.publish(cmd)
            return
        if self.state in ('backup', 'turn'):
            self.run_recovery(cmd)
            self.cmd_vel_pub.publish(cmd)
            return
        if self.state == 'look':
            self.run_look(cmd)
            self.cmd_vel_pub.publish(cmd)
            return

        tx, ty = self.waypoints[self.wp_index]
        dist = math.hypot(tx - self.x, ty - self.y)
        heading_err = wrap_angle(math.atan2(ty - self.y, tx - self.x) - self.yaw)

        target = tuple(self.waypoints[self.wp_index])
        tol = self.WAYPOINT_TOL if (target in self.looks or target == self.fire_goal) else self.PASS_TOL
        if dist < tol:
            if self.fire_state == 'approach' and target == self.fire_goal:
                self.next_waypoint('reached (big fire)')
                self.arrive_at_fire()
            elif target in self.looks and target not in self.looked:
                self.looked.add(target)
                self.look_queue = list(self.looks[target])
                self.look_dwell_start = None
                self.state, self.state_start = 'look', self.now()
            else:
                self.next_waypoint('reached')
        elif self.now() - self.wp_start_time > self.WAYPOINT_TIMEOUT:
            self.next_waypoint('skipped (timeout)')
        elif not self.check_progress():
            self.start_recovery('No progress')
        elif abs(heading_err) > self.TURN_IN_PLACE_ANGLE:
            if self.front_half_min < 0.15 and self.rear_min > 0.20:
                cmd.linear.x = -0.05
            cmd.angular.z = max(min(1.5 * heading_err, 0.8), -0.8)
        elif self.corridor_min < self.STOP_DIST and self.corridor_min < dist:
            self.start_recovery(f'Obstacle {self.corridor_min:.2f} m ahead', obstacle_ahead=True)
        else:
            speed = self.MAX_SPEED if self.corridor_min > self.SLOW_DIST else self.SLOW_SPEED
            speed *= max(0.3, 1.0 - abs(heading_err) / max(self.TURN_IN_PLACE_ANGLE, 0.4))
            cmd.linear.x = min(speed, 0.5 * dist + 0.05)
            cmd.angular.z = max(min(1.5 * heading_err, 1.0), -1.0)
            if self.left_min < self.SIDE_PUSH_DIST:
                cmd.angular.z -= 0.4
            elif self.right_min < self.SIDE_PUSH_DIST:
                cmd.angular.z += 0.4
        self.cmd_vel_pub.publish(cmd)

    def publish_viz(self):
        #sender roadmap, rute og vegger til rviz
        if self.prm is not None:
            self.publish_prm_viz()
            self.publish_path_viz()
            self.publish_walls_viz()

    def publish_walls_viz(self):
        #vegger som er sett i grønt og vegger som ikke er sett i rødt
        #må noen ganger som inn for at det skal komme opp
        if not hasattr(self.prm, 'seen'):
            return
        arr = MarkerArray()
        for i, (name, mask, rgb) in enumerate([('seen', self.prm.seen, (0.1, 0.9, 0.1)),
                                                ('unseen', ~self.prm.seen, (0.95, 0.1, 0.1))]):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = f'{self.ns}_walls_{name}', i
            m.type, m.action = Marker.POINTS, Marker.ADD
            m.scale.x = m.scale.y = 0.05
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 0.9
            for x, y in self.prm.wall_xy[mask]:
                m.points.append(Point(x=float(x), y=float(y), z=0.05))
            arr.markers.append(m)
        self.walls_pub.publish(arr)

    def publish_prm_viz(self):
        #tegner roadmap til rviz
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        nodes = Marker()
        nodes.header.frame_id = 'map'
        nodes.header.stamp = stamp
        nodes.ns, nodes.id = f'{self.ns}_prm_nodes', 0
        nodes.type, nodes.action = Marker.POINTS, Marker.ADD
        nodes.scale.x = nodes.scale.y = 0.05
        nodes.color.g, nodes.color.b, nodes.color.a = 0.5, 1.0, 0.8
        for px, py in self.prm.nodes:
            wx, wy = self.prm._pixel_to_world(px, py)
            nodes.points.append(Point(x=float(wx), y=float(wy)))
        arr.markers.append(nodes)

        edges = Marker()
        edges.header.frame_id = 'map'
        edges.header.stamp = stamp
        edges.ns, edges.id = f'{self.ns}_prm_edges', 1
        edges.type, edges.action = Marker.LINE_LIST, Marker.ADD
        edges.scale.x = 0.01
        edges.color.r = edges.color.g = edges.color.b = 0.5
        edges.color.a = 0.4
        for u in self.prm.graph:
            w1 = self.prm._pixel_to_world(*self.prm.nodes[u])
            for v in self.prm.graph[u]:
                if u < v:
                    w2 = self.prm._pixel_to_world(*self.prm.nodes[v])
                    edges.points.append(Point(x=float(w1[0]), y=float(w1[1])))
                    edges.points.append(Point(x=float(w2[0]), y=float(w2[1])))
        arr.markers.append(edges)
        self.prm_pub.publish(arr)

    def publish_path_viz(self):
        #tegner ruten til roboten i rvizs
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for wx, wy in self.waypoints[self.wp_index:]:
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = float(wx)
            p.pose.position.y = float(wy)
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()