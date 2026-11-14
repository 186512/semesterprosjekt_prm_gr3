import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from ros2_aruco_interfaces.msg import ArucoMarkers
from scoring_interfaces.srv import SetMarkerPosition

ROBOTS = ['tb3_0', 'tb3_1']
RESCUE = 'tb3_2'
BIG_FIRE_ID = 4


class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        self.declare_parameter('max_range', 2.5)
        # Hvor langt ut fra veggen (mot finneren) tb3_2 sitt mål legges.
        self.declare_parameter('approach_offset', 0.6)
        # Hvis tb3_2 aldri kommer frem, slippes finneren etter så mange sekunder,
        # så resten av dens halvdel av bygningen ikke blir liggende uutforsket.
        self.declare_parameter('fire_max_wait', 150.0)
        self.max_range = self.get_parameter('max_range').value
        self.approach_offset = self.get_parameter('approach_offset').value
        self.fire_max_wait = self.get_parameter('fire_max_wait').value
        self.fire_found_at = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.client = self.create_client(SetMarkerPosition, '/set_marker_position')
        self.yield_pub = self.create_publisher(Bool, '/tb3_1/yield', 10)
        self.is_yielding = False

        self.fire_target_pub = self.create_publisher(Point, f'/{RESCUE}/fire_target', 10)
        # Selve markørposisjonen: tb3_2 regner seg som fremme når den er innenfor 1.6 m av den
        self.fire_marker_pub = self.create_publisher(Point, f'/{RESCUE}/fire_marker', 10)
        self.wait_pubs = {
            ns: self.create_publisher(Bool, f'/{ns}/wait_at_fire', 10) for ns in ROBOTS
        }
        # Finneren får beskjed om å kjøre inn til brannen (PRM-explorer gjør selve kjøringen),
        # og sier fra på /<ns>/at_fire når den står innenfor 2 m-kravet.
        self.approach_pubs = {
            ns: self.create_publisher(Point, f'/{ns}/fire_approach', 10) for ns in ROBOTS
        }
        self.finder_at_fire = False
        for ns in ROBOTS:
            self.create_subscription(
                Bool, f'/{ns}/at_fire', lambda msg, ns=ns: self.at_fire_callback(ns, msg), 10)
        self.seen_logged = set()      # (robot, id) som allerede er logget som sett
        self.tf_warned = set()

        # Tilstand for stor brann
        self.fire_found = False       # ID 4 er sett av en av wall followerne
        self.finder_id = None         # hvem som fant den (tb3_0 / tb3_1)
        self.fire_marker_pos = None   # markørens posisjon i map
        self.rescue_goal = None       # punkt foran markøren, på finnerens side
        self.rescue_arrived = False   # tb3_2 er fremme
        self.release_at = None        # tidspunkt finneren slippes fri
        self.fire_done = False        # alt ferdig, finneren kjører videre

        self.found = set()
        self.in_flight = set()
        self.pending = {}
        self.retry_after = {}

        for ns in ROBOTS:
            self.create_subscription(
                ArucoMarkers, '/' + ns + '/aruco_markers',
                lambda msg, ns=ns: self.markers_callback(ns, msg), 10)

        # tb3_2 sier selv fra når den er fremme (riktig side av veggen)
        self.create_subscription(
            Bool, f'/{RESCUE}/rescue_arrived', self.rescue_arrived_callback, 10)

        self.create_timer(0.5, self.timer_callback)
        self.create_timer(0.1, self.proximity_callback)

    # ------------------------------------------------------------------
    def proximity_callback(self):
        self.handle_yield()
        self.handle_big_fire()

    def handle_yield(self):
        try:
            tf0 = self.tf_buffer.lookup_transform('map', 'tb3_0/base_footprint', Time())
            tf1 = self.tf_buffer.lookup_transform('map', 'tb3_1/base_footprint', Time())
        except TransformException:
            return

        d = math.hypot(tf0.transform.translation.x - tf1.transform.translation.x,
                       tf0.transform.translation.y - tf1.transform.translation.y)

        if d < 0.70 and not self.is_yielding:
            self.is_yielding = True
            self.yield_pub.publish(Bool(data=True))
        elif d > 0.95 and self.is_yielding:
            self.is_yielding = False
            self.yield_pub.publish(Bool(data=False))

    def handle_big_fire(self):
        if not self.fire_found or self.fire_done:
            return

        # Finneren står stille helt til vi slipper den
        self.wait_pubs[self.finder_id].publish(Bool(data=True))
        if not self.finder_at_fire:
            # Gjenta til finneren har kommet seg innenfor rekkevidde av brannen
            self.approach_pubs[self.finder_id].publish(self.fire_marker_pos)

        if not self.rescue_arrived:
            # Publiser målet jevnlig, så tb3_2 får det selv om den startet sent
            self.fire_marker_pub.publish(self.fire_marker_pos)
            self.fire_target_pub.publish(self.rescue_goal)

        now = self.get_clock().now().nanoseconds
        if self.fire_found_at is not None and now - self.fire_found_at > self.fire_max_wait * 1e9:
            self.fire_done = True
            self.wait_pubs[self.finder_id].publish(Bool(data=False))
            self.get_logger().warn(
                f'{RESCUE} kom ikke frem på {self.fire_max_wait:.0f} s - '
                f'slipper {self.finder_id} så den kan lete videre.')
            return

        # Slipp finneren først når ID 4 er akseptert OG tb3_2 er fremme.
        # Vent litt ekstra så scoring-noden (sjekker hvert 0.5 s) rekker å
        # registrere at begge robotene er nær brannen (+300 poeng).
        if self.rescue_arrived and self.finder_at_fire and BIG_FIRE_ID in self.found:
            if self.release_at is None:
                self.release_at = now + 5 * 10 ** 9
            elif now >= self.release_at:
                self.fire_done = True
                self.wait_pubs[self.finder_id].publish(Bool(data=False))
                self.get_logger().info(
                    f'Stor brann håndtert. {self.finder_id} fortsetter å lete.')

    def at_fire_callback(self, ns, msg):
        if msg.data and self.fire_found and ns == self.finder_id and not self.finder_at_fire:
            self.finder_at_fire = True
            self.get_logger().info(f'{ns} står ved den store brannen og venter på {RESCUE}.')

    def rescue_arrived_callback(self, msg):
        if msg.data and self.fire_found and not self.rescue_arrived:
            self.rescue_arrived = True
            self.get_logger().info(f'{RESCUE} HAR ANKOMMET BRANNEN!')

    def compute_rescue_goal(self, ns, mx, my):
        """Punkt approach_offset meter ut fra markøren, i retning finneren.
        Linja markør -> finner er kameraets siktlinje, så den krysser ingen vegg,
        og punktet havner derfor alltid på samme side av veggen som finneren."""
        try:
            tf_f = self.tf_buffer.lookup_transform('map', f'{ns}/base_footprint', Time())
        except TransformException:
            self.get_logger().warn('Fant ikke finnerens posisjon, bruker markøren som mål')
            return Point(x=mx, y=my, z=0.0)

        fx = tf_f.transform.translation.x
        fy = tf_f.transform.translation.y
        d = math.hypot(fx - mx, fy - my)
        if d < 1e-3:
            return Point(x=fx, y=fy, z=0.0)
        offset = min(self.approach_offset, 0.8 * d)
        return Point(x=mx + offset * (fx - mx) / d,
                     y=my + offset * (fy - my) / d,
                     z=0.0)

    # ------------------------------------------------------------------
    def markers_callback(self, ns, msg):
        now = self.get_clock().now().nanoseconds
        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            marker_id = int(marker_id)
            if marker_id not in range(5):
                continue
            if marker_id in self.found or marker_id in self.in_flight:
                continue
            if now < self.retry_after.get(marker_id, 0):
                continue
            if (ns, marker_id) not in self.seen_logged:
                self.seen_logged.add((ns, marker_id))
                self.get_logger().info(f'{ns} ser ID {marker_id} ({pose.position.z:.1f} m unna)')
            if pose.position.z > self.max_range:
                continue
            # ID 4 rapporteres uansett avstand (<= max_range); finneren kjøres inn etterpå.
            self.pending.setdefault(marker_id, (ns, msg.header.stamp, pose))

    def timer_callback(self):
        for marker_id, (ns, stamp, pose) in list(self.pending.items()):
            frame = ns + '/camera_rgb_optical_frame'

            try:
                tf = self.tf_buffer.lookup_transform('map', frame, Time.from_msg(stamp))
            except TransformException as e:
                age = self.get_clock().now().nanoseconds - Time.from_msg(stamp).nanoseconds
                if marker_id not in self.tf_warned:
                    self.tf_warned.add(marker_id)
                    self.get_logger().warn(f'TF map -> {frame} feilet for ID {marker_id}: {e}')
                if age > 3 * 10 ** 9:
                    del self.pending[marker_id]
                    self.get_logger().warn(f'Ga opp ID {marker_id} (ingen TF på 3 s), venter på ny observasjon')
                continue

            if not self.client.service_is_ready():
                self.get_logger().warn('Scoring-tjenesten /set_marker_position er ikke oppe ennå',
                                       throttle_duration_sec=5.0)
                continue

            map_pose = tf2_geometry_msgs.do_transform_pose(pose, tf)
            x = map_pose.position.x
            y = map_pose.position.y

            if marker_id == BIG_FIRE_ID and not self.fire_found:
                self.fire_found = True
                self.fire_found_at = self.get_clock().now().nanoseconds
                self.finder_id = ns
                self.fire_marker_pos = Point(x=x, y=y, z=0.0)
                self.rescue_goal = self.compute_rescue_goal(ns, x, y)
                self.wait_pubs[ns].publish(Bool(data=True))
                self.approach_pubs[ns].publish(self.fire_marker_pos)
                self.fire_target_pub.publish(self.rescue_goal)
                self.get_logger().info(
                    f'STOR BRANN FUNNET AV {ns} ved ({x:.2f}, {y:.2f})! '
                    f'Sender {RESCUE} til ({self.rescue_goal.x:.2f}, '
                    f'{self.rescue_goal.y:.2f}).')

            # Alle markører (også ID 4) rapporteres med en gang
            self.get_logger().info(f'Rapporterer ID {marker_id} ved ({x:.2f}, {y:.2f}) til scoring')
            req = SetMarkerPosition.Request()
            req.marker_id = marker_id
            req.marker_position = Point(x=x, y=y, z=0.0)

            del self.pending[marker_id]
            self.in_flight.add(marker_id)
            future = self.client.call_async(req)
            future.add_done_callback(lambda f, i=marker_id: self.response_callback(i, f))

    def response_callback(self, marker_id, future):
        self.in_flight.discard(marker_id)
        try:
            accepted = future.result().accepted
        except Exception as e:
            self.get_logger().error('Service-kall feilet for ID %d: %s' % (marker_id, str(e)))
            accepted = False

        if accepted:
            self.found.add(marker_id)
            self.get_logger().info('ID %d akseptert av scoring (%d/5 funnet)'
                                   % (marker_id, len(self.found)))
        else:
            self.retry_after[marker_id] = self.get_clock().now().nanoseconds + 2 * 10 ** 9
            self.get_logger().warn('ID %d IKKE akseptert - tar ny observasjon. Se scoring-terminalen.'
                                   % marker_id)


def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
