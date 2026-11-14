#!/usr/bin/env python3
"""
Probabilistic Roadmap (PRM) for multi-robot search & rescue exploration.

Pipeline
  1. Occupancy grid  -> free-space mask, obstacles inflated by the robot radius
  2. Flood fill from the robots' start positions -> only the part of the building
     the robots can actually reach (the empty area outside the walls is ignored)
  3. Random samples inside that region (count scales with the area), connected to
     their k nearest neighbours with straight collision-free edges
  4. Extra samples are added until the roadmap is one connected piece
  5. Sparse coverage goals ~1 m apart are split between the robots (k-means),
     each robot visits its goals nearest-first, legs routed with Dijkstra / A*
     and shortcut-smoothed

Pixel convention: (px, py) with py = 0 at the TOP of the image, like the map .png.
"""
import heapq
import itertools
import math
import os
import random

import cv2
import numpy as np
import yaml


class PRMGenerator:
    def __init__(self, free_mask, resolution, origin_x, origin_y,
                 robot_radius=0.24, k=12, max_edge_length=2.0,
                 node_density=4.0, seed=42):
        """
        free_mask: uint8 image (row 0 = top), 255 = free, 0 = occupied/unknown.
        node_density: roadmap nodes per square metre of reachable free space.
        """
        self.raw_free = free_mask
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.robot_radius = robot_radius
        self.k = k
        self.max_edge_length = max_edge_length
        self.node_density = node_density
        self.rng = random.Random(seed)

        r = max(1, int(math.ceil(robot_radius / self.resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.map_data = cv2.erode(self.raw_free, kernel)   # inflated free space

        self.region = None   # reachable inflated free space
        self.nodes = []      # list of (px, py)
        self.graph = {}      # node index -> list of neighbour indices

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_yaml(cls, yaml_path, **kw):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Could not load map image {img_path}')
        p = img.astype(np.float32) / 255.0
        occ = p if int(meta.get('negate', 0)) else 1.0 - p
        free = ((occ < float(meta.get('free_thresh', 0.196))).astype(np.uint8)) * 255
        return cls(free, meta['resolution'], meta['origin'][0], meta['origin'][1], **kw)

    @classmethod
    def from_occupancy_grid(cls, msg, **kw):
        """Build from a nav_msgs/OccupancyGrid (e.g. the /map topic)."""
        w, h = msg.info.width, msg.info.height
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        free = ((data >= 0) & (data < 50)).astype(np.uint8) * 255
        free = np.flipud(free).copy()   # grid row 0 is the bottom; image row 0 is the top
        o = msg.info.origin.position
        return cls(free, msg.info.resolution, o.x, o.y, **kw)

    # ------------------------------------------------------------ coordinates
    def _pixel_to_world(self, px, py):
        wx = (px + 0.5) * self.resolution + self.origin_x
        wy = (self.map_data.shape[0] - py - 0.5) * self.resolution + self.origin_y
        return (wx, wy)

    def _world_to_pixel(self, wx, wy):
        px = int((wx - self.origin_x) / self.resolution)
        py = int(self.map_data.shape[0] - (wy - self.origin_y) / self.resolution)
        return (px, py)

    def is_free_world(self, wx, wy, inflated=False):
        return self._is_free(*self._world_to_pixel(wx, wy),
                             self.map_data if inflated else self.raw_free)

    # ------------------------------------------------------- collision checks
    def _is_free(self, px, py, mask=None):
        mask = self.map_data if mask is None else mask
        return 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and mask[py, px] > 0

    def _check_collision(self, p1, p2, mask=None):
        """True if the straight line p1->p2 (pixels) touches a blocked pixel."""
        mask = self.map_data if mask is None else mask
        n = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1])) + 2
        xs = np.rint(np.linspace(p1[0], p2[0], n)).astype(int)
        ys = np.rint(np.linspace(p1[1], p2[1], n)).astype(int)
        if xs.min() < 0 or ys.min() < 0 or xs.max() >= mask.shape[1] or ys.max() >= mask.shape[0]:
            return True
        return bool(np.any(mask[ys, xs] == 0))

    def _dist(self, i, j):
        (x1, y1), (x2, y2) = self.nodes[i], self.nodes[j]
        return math.hypot(x2 - x1, y2 - y1)

    def _snap_free(self, px, py, mask, max_r_m=0.6):
        """Nearest free pixel of mask to (px, py), searching up to max_r_m metres."""
        if self._is_free(px, py, mask):
            return px, py
        r = int(max_r_m / self.resolution)
        y0, y1 = max(0, py - r), min(mask.shape[0], py + r + 1)
        x0, x1 = max(0, px - r), min(mask.shape[1], px + r + 1)
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if len(xs) == 0:
            return None
        i = int(np.argmin((xs + x0 - px) ** 2 + (ys + y0 - py) ** 2))
        return int(xs[i] + x0), int(ys[i] + y0)

    # ---------------------------------------------------------------- roadmap
    def generate_roadmap(self, start_points_world):
        """Build the roadmap over the free space reachable from the given start points."""
        # 1) reachable region = flood fill of the inflated free space from each start
        h, w = self.map_data.shape
        self.region = np.zeros_like(self.map_data)
        for sx, sy in start_points_world:
            seed = self._snap_free(*self._world_to_pixel(sx, sy), self.map_data)
            if seed is None:
                raise RuntimeError(f'Start ({sx:.2f}, {sy:.2f}) is not near free space - '
                                   f'is the robot pose in the map frame?')
            if self.region[seed[1], seed[0]]:
                continue
            ff = self.map_data.copy()
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, mask, seed, 128)
            self.region[ff == 128] = 255

        ys, xs = np.nonzero(self.region)
        area = len(xs) * self.resolution ** 2
        n = int(np.clip(area * self.node_density, 200, 3000))
        self.region_area = area

        # 2) sample + connect, adding samples until the roadmap is one piece
        self.nodes = []
        self.graph = {}
        for _ in range(4):
            idx = self.rng.sample(range(len(xs)), min(n, len(xs)))
            self._add_nodes([(int(xs[i]), int(ys[i])) for i in idx])
            comps = self.components()
            largest = max(len(c) for c in comps)
            if largest >= 0.97 * len(self.nodes):
                break
            n = max(200, n // 2)
        self.graph = {i: sorted(v) for i, v in self.graph.items()}

    def _add_nodes(self, new_nodes):
        start = len(self.nodes)
        self.nodes.extend(new_nodes)
        for i in range(start, len(self.nodes)):
            self.graph[i] = set()
        pts = np.array(self.nodes, dtype=np.float32)
        max_px = self.max_edge_length / self.resolution
        for i in range(start, len(self.nodes)):
            d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
            for j in np.argsort(d)[1:self.k + 1]:
                j = int(j)
                if d[j] > max_px or j in self.graph[i]:
                    continue
                if not self._check_collision(self.nodes[i], self.nodes[j]):
                    self.graph[i].add(j)
                    self.graph[j].add(i)

    def components(self):
        left, comps = set(self.graph), []
        while left:
            c = self._component(next(iter(left)))
            comps.append(c)
            left -= c
        return comps

    def _component(self, start):
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v in self.graph[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    # ------------------------------------------------------ dynamic obstacles
    def add_obstacle(self, wx, wy, radius=0.25):
        """
        Block an obstacle the lidar sees but the blueprint does not show.
        Returns the number of roadmap edges removed (0 if it is a known wall).
        """
        px, py = self._world_to_pixel(wx, wy)
        # Close to a wall that is already on the map? Then it is not new debris.
        if self._snap_free(px, py, 255 - self.raw_free,
                           max_r_m=self.robot_radius + 0.15) is not None:
            return 0
        r_raw = max(1, int(math.ceil(radius / self.resolution)))
        r_inf = int(math.ceil((radius + self.robot_radius) / self.resolution))
        cv2.circle(self.raw_free, (px, py), r_raw, 0, -1)
        cv2.circle(self.map_data, (px, py), r_inf, 0, -1)
        reach = self.max_edge_length / self.resolution + r_inf
        removed = 0
        for u, (ux, uy) in enumerate(self.nodes):
            if math.hypot(ux - px, uy - py) > reach:
                continue
            keep = []
            for v in self.graph[u]:
                if self._check_collision(self.nodes[u], self.nodes[v]):
                    removed += 1
                    self.graph[v] = [n for n in self.graph[v] if n != u]
                else:
                    keep.append(v)
            self.graph[u] = keep
        return removed

    # ----------------------------------------------------------- graph search
    def nearest_node(self, wx, wy):
        """Closest connected roadmap node with line of sight to (wx, wy)."""
        px, py = self._world_to_pixel(wx, wy)
        usable = [i for i in range(len(self.nodes)) if self.graph[i] and self._is_free(*self.nodes[i])]
        order = sorted(usable or range(len(self.nodes)),
                       key=lambda i: (self.nodes[i][0] - px) ** 2 + (self.nodes[i][1] - py) ** 2)
        for i in order[:40]:
            if not self._check_collision((px, py), self.nodes[i], self.raw_free):
                return i
        return order[0]

    def dijkstra(self, source):
        cost, prev, pq = {source: 0.0}, {source: None}, [(0.0, source)]
        while pq:
            c, u = heapq.heappop(pq)
            if c > cost[u]:
                continue
            for v in self.graph[u]:
                nc = c + self._dist(u, v)
                if nc < cost.get(v, math.inf):
                    cost[v], prev[v] = nc, u
                    heapq.heappush(pq, (nc, v))
        return cost, prev

    @staticmethod
    def _seg_clear(p1, p2, avoid_px):
        """True if segment p1-p2 (pixels) stays outside every (cx, cy, r) circle in avoid_px."""
        if not avoid_px:
            return True
        x1, y1 = p1
        dx, dy = p2[0] - x1, p2[1] - y1
        L2 = dx * dx + dy * dy
        for cx, cy, r in avoid_px:
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / L2))
            if math.hypot(x1 + t * dx - cx, y1 + t * dy - cy) < r:
                return False
        return True

    def astar(self, start, goal, avoid_px=None):
        """Node-index path start->goal, or None if unreachable.
        avoid_px: temporary obstacles (cx, cy, r) in pixels, e.g. other robots."""
        cost, prev = {start: 0.0}, {start: None}
        pq = [(self._dist(start, goal), 0.0, start)]
        while pq:
            _, c, u = heapq.heappop(pq)
            if u == goal:
                return self._reconstruct(prev, goal)
            if c > cost[u]:
                continue
            for v in self.graph[u]:
                if avoid_px and not self._seg_clear(self.nodes[u], self.nodes[v], avoid_px):
                    continue
                nc = c + self._dist(u, v)
                if nc < cost.get(v, math.inf):
                    cost[v], prev[v] = nc, u
                    heapq.heappush(pq, (nc + self._dist(v, goal), nc, v))
        return None

    @staticmethod
    def _reconstruct(prev, goal):
        path, n = [], goal
        while n is not None:
            path.append(n)
            n = prev[n]
        return path[::-1]

    def _smooth(self, node_path, avoid_px=None):
        """Shortcut a node path: skip intermediate nodes with clear line of sight."""
        pts = [self.nodes[i] for i in node_path]
        if len(pts) <= 2:
            return pts
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and (self._check_collision(pts[i], pts[j])
                                 or not self._seg_clear(pts[i], pts[j], avoid_px)):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def plan_path_world(self, start_xy, goal_xy, avoid=None):
        """
        World-coordinate path between two points (e.g. drive to the big fire).
        avoid: list of (x, y, radius) in metres to keep away from, e.g. other robots.
        Returns [] if there is no route.
        """
        avoid_px = []
        for ax, ay, r in (avoid or []):
            px, py = self._world_to_pixel(ax, ay)
            avoid_px.append((px, py, r / self.resolution))
        s, g = self.nearest_node(*start_xy), self.nearest_node(*goal_xy)
        # start/goal nodes themselves may sit inside an avoid circle: only edges are checked
        path = self.astar(s, g, avoid_px)
        if path is None:
            return []
        pts = self._smooth(path, avoid_px)
        if avoid_px and not self._seg_clear(self._world_to_pixel(*start_xy), pts[0], avoid_px):
            # the hop from where we stand to the first node would hit a robot
            if len(pts) > 1 and self._seg_clear(self._world_to_pixel(*start_xy), pts[1], avoid_px) \
                    and not self._check_collision(self._world_to_pixel(*start_xy), pts[1]):
                pts = pts[1:]
        world = [self._pixel_to_world(*p) for p in pts]
        world.append(tuple(goal_xy))
        return world

    # ------------------------------------------------------------ exploration
    def _coverage_goals(self, nodes, spacing_px):
        """Sparse subset of nodes: no two goals closer than spacing_px."""
        order = sorted(nodes)
        self.rng.shuffle(order)
        goals = []
        for i in order:
            if all(self._dist(i, g) >= spacing_px for g in goals):
                goals.append(i)
        return sorted(goals)

    def _split_goals(self, goals, starts_px):
        """k-means the goals into one cluster per robot; give each robot the nearest cluster."""
        n_robots = len(starts_px)
        if n_robots <= 1 or len(goals) < n_robots:
            return [goals] + [[] for _ in range(n_robots - 1)]
        pts = np.array([self.nodes[g] for g in goals], dtype=float)
        centers = [pts[0]]
        while len(centers) < n_robots:
            d = np.min([np.hypot(*(pts - c).T) for c in centers], axis=0)
            centers.append(pts[int(np.argmax(d))])
        centers = np.array(centers)
        for _ in range(30):
            labels = np.argmin(np.stack([np.hypot(*(pts - c).T) for c in centers]), axis=0)
            centers = np.array([pts[labels == c].mean(axis=0) if np.any(labels == c) else centers[c]
                                for c in range(n_robots)])
        clusters = [[g for g, l in zip(goals, labels) if l == c] for c in range(n_robots)]
        best, best_cost = None, math.inf
        for perm in itertools.permutations(range(n_robots)):
            cost = sum(math.hypot(*(centers[perm[r]] - np.array(starts_px[r])))
                       for r in range(n_robots))
            if cost < best_cost:
                best, best_cost = perm, cost
        return [clusters[best[r]] for r in range(n_robots)]

    def _order_goals(self, start, goals):
        """Visiting order: nearest-neighbour tour, then improved with 2-opt on roadmap distances."""
        stops = [start] + list(goals)
        sp = {s: self.dijkstra(s) for s in stops}            # shortest paths from every stop
        stops = [s for s in stops if s == start or s in sp[start][0]]
        D = {(a, b): sp[a][0].get(b, math.inf) for a in stops for b in stops}

        tour, left = [start], set(stops[1:])
        while left:
            nxt = min(left, key=lambda g: D[tour[-1], g])
            tour.append(nxt)
            left.discard(nxt)

        def length(t):
            return sum(D[a, b] for a, b in zip(t, t[1:]))
        improved = True
        while improved:   # 2-opt for an open path with a fixed start
            improved = False
            for i in range(1, len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    if length(cand) < length(tour) - 1e-6:
                        tour, improved = cand, True
        return tour, sp

    # ------------------------------------------------------- camera visibility
    # Markers hang on walls, and the camera only sees ~60 deg ahead and the coordinator
    # only accepts detections closer than 2.5 m. So the planner picks viewpoints from which
    # every wall is within camera range, and tells the robot which way to look at each one.
    VIS_RES = 0.05                    # metres per cell of the coarse visibility grid
    CAM_RANGE = 2.2                   # plan to see walls closer than this
    CAM_FOV = math.radians(56)        # a bit narrower than the real 62 deg
    LOOK_SECTOR = math.radians(50)    # width of one "look" when turning on the spot
    MAX_INCIDENCE = math.radians(60)  # ArUco fails when the wall is viewed too obliquely

    def _build_visibility(self):
        f = int(round(self.VIS_RES / self.resolution))
        h, w = self.raw_free.shape[0] // f, self.raw_free.shape[1] // f
        occ_frac = cv2.resize((self.raw_free[:h * f, :w * f] == 0).astype(np.float32),
                              (w, h), interpolation=cv2.INTER_AREA)
        self.vis_occ = occ_frac > 0.2                       # coarse obstacle grid
        reach = cv2.resize((self.region[:h * f, :w * f] > 0).astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_AREA) > 0
        k = int(math.ceil((self.robot_radius + 0.1) / self.VIS_RES))
        near = cv2.dilate(reach.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        room = (near > 0) & ~self.vis_occ                   # free space next to where robots drive
        wall = self.vis_occ & (cv2.dilate(room.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        self.wall_id = np.full(self.vis_occ.shape, -1, np.int32)
        wy, wx = np.nonzero(wall)
        self.wall_id[wy, wx] = np.arange(len(wx))
        self.wall_xy = np.stack([(wx + 0.5) * self.VIS_RES + self.origin_x,
                                 (self.vis_occ.shape[0] - wy - 0.5) * self.VIS_RES + self.origin_y], 1)
        # Wall normals (pointing into the room): average direction to nearby free cells
        r = 3
        offs = np.arange(-r, r + 1, dtype=np.float32)
        kx = np.tile(offs, (2 * r + 1, 1))
        roomf = room.astype(np.float32)
        nx = cv2.filter2D(roomf, -1, kx, borderType=cv2.BORDER_CONSTANT)[wy, wx]
        ny = -cv2.filter2D(roomf, -1, kx.T, borderType=cv2.BORDER_CONSTANT)[wy, wx]
        nn = np.hypot(nx, ny) + 1e-9
        self.wall_normal = np.stack([nx / nn, ny / nn], 1)
        self.seen = np.zeros(len(wx), bool)                 # walls the camera has actually seen

    def _cast(self, wx, wy, angles, max_range):
        """Wall-cell ids hit first by rays from (wx, wy) at the given angles (world frame)."""
        H, W = self.vis_occ.shape
        cx = (wx - self.origin_x) / self.VIS_RES
        cy = H - (wy - self.origin_y) / self.VIS_RES
        steps = np.arange(1.0, max_range / self.VIS_RES, 0.5)
        xs = (cx + np.outer(np.cos(angles), steps)).astype(int)
        ys = (cy - np.outer(np.sin(angles), steps)).astype(int)
        inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = np.clip(xs, 0, W - 1), np.clip(ys, 0, H - 1)
        hit = ~inb | self.vis_occ[ys, xs]
        any_hit = hit.any(1)
        first = hit.argmax(1)
        r = np.nonzero(any_hit)[0]
        ids = self.wall_id[ys[r, first[r]], xs[r, first[r]]]
        ok = ids >= 0
        ids, ra = ids[ok], np.asarray(angles)[r[ok]]
        # keep only walls seen face-on enough for ArUco detection
        facing = -(np.cos(ra) * self.wall_normal[ids, 0] + np.sin(ra) * self.wall_normal[ids, 1])
        return np.unique(ids[facing > math.cos(self.MAX_INCIDENCE)])

    def camera_view(self, wx, wy, yaw, n_rays=40):
        """Wall cells the camera sees from this pose (marks them as seen)."""
        a = yaw + np.linspace(-self.CAM_FOV / 2, self.CAM_FOV / 2, n_rays)
        ids = self._cast(wx, wy, a, self.CAM_RANGE)
        self.seen[ids] = True
        return ids

    def _look_headings(self, wx, wy, ids):
        """Group wall cells around a viewpoint into as few camera headings as possible."""
        if len(ids) == 0:
            return []
        p = self.wall_xy[ids]
        ang = np.arctan2(p[:, 1] - wy, p[:, 0] - wx)
        order = np.argsort(ang)
        a, ids = ang[order], ids[order]
        gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
        start = (int(np.argmax(gaps)) + 1) % len(a)           # begin after the biggest gap
        a = np.unwrap(np.roll(a, -start))
        ids = np.roll(ids, -start)
        looks, i = [], 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[j + 1] - a[i] <= self.LOOK_SECTOR:
                j += 1
            if j - i + 1 >= 3:                                # ignore slivers of a few cells
                looks.append((float(math.atan2(math.sin((a[i] + a[j]) / 2), math.cos((a[i] + a[j]) / 2))),
                              ids[i:j + 1].copy()))
            i = j + 1
        return looks

    def _viewpoints(self, reachable, max_candidates=900, min_gain=6):
        """Greedy set cover: fewest roadmap nodes that together see (almost) every wall."""
        cand = sorted(reachable)
        if len(cand) > max_candidates:
            cand = sorted(self.rng.sample(cand, max_candidates))
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        views = {c: set(self._cast(*self._pixel_to_world(*self.nodes[c]), angles,
                                   self.CAM_RANGE).tolist()) for c in cand}
        coverable = set().union(*views.values()) if views else set()
        covered, chosen = set(), []
        while len(covered) < 0.98 * len(coverable):
            best = max(cand, key=lambda c: len(views[c] - covered))
            new = views[best] - covered
            if len(new) < min_gain:
                break
            chosen.append((best, np.array(sorted(new), dtype=np.int64)))
            covered |= new
        self.coverable = len(coverable)
        return chosen

    def get_exploration_plan(self, robot_starts, robot_index=0):
        """
        Plan for one robot.
        Returns (waypoints, looks):
          waypoints: list of (x, y) map-frame points to drive through
          looks:     {(x, y) viewpoint: [(heading, wall ids), ...]} - where to turn and look
        robot_starts: (x, y) start of every explorer, same order on all robots.
        """
        start_nodes = [self.nearest_node(*s) for s in robot_starts]
        reachable = set().union(*(self._component(s) for s in start_nodes))
        self._build_visibility()
        chosen = self._viewpoints(reachable)
        new_cells = {g: ids for g, ids in chosen}
        my_goals = self._split_goals([g for g, _ in chosen],
                                     [self.nodes[s] for s in start_nodes])[robot_index]

        tour, sp = self._order_goals(start_nodes[robot_index], my_goals)
        route_px = [self.nodes[tour[0]]]
        for a, b in zip(tour, tour[1:]):
            route_px.extend(self._smooth(self._reconstruct(sp[a][1], b))[1:])
        waypoints = [self._pixel_to_world(px, py) for px, py in route_px]

        looks = {}
        for g in tour[1:]:
            wxy = self._pixel_to_world(*self.nodes[g])
            looks[wxy] = self._look_headings(*wxy, new_cells[g])
        return waypoints, looks

    def get_exploration_path(self, robot_starts, robot_index=0):
        return self.get_exploration_plan(robot_starts, robot_index)[0]