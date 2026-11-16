#!/usr/bin/env python3
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
        self.map_data = cv2.erode(self.raw_free, kernel)   

        self.region = None   
        self.nodes = []  
        self.graph = {}   



    #henter yaml filen for kartene. finner ut hvor mange meter en piskel er. og laster opp bildet med gråtone
    #hvor hver pixel har en verdi 0(svart) til hvit(255). verdier under free_thresh betyr fritt og verdier over betyr vegg.
    @classmethod
    def from_yaml(cls, yaml_path, **kw): 
        with open(yaml_path) as f: 
            meta = yaml.safe_load(f)
        img_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'could not load image {img_path}')
        p = img.astype(np.float32) / 255.0
        occ = p if int(meta.get('negate', 0)) else 1.0 - p
        free = ((occ < float(meta.get('free_thresh', 0.196))).astype(np.uint8)) * 255
        return cls(free, meta['resolution'], meta['origin'][0], meta['origin'][1], **kw)

    #henter /map topic. det er denne robotene bruker.
    @classmethod
    def from_occupancy_grid(cls, msg, **kw):
        w, h = msg.info.width, msg.info.height
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        free = ((data >= 0) & (data < 50)).astype(np.uint8) * 255
        free = np.flipud(free).copy() 
        o = msg.info.origin.position
        return cls(free, msg.info.resolution, o.x, o.y, **kw)

    
    #gjør om en position i kartet bildet til meter
    def _pixel_to_world(self, px, py):
        wx = (px + 0.5) * self.resolution + self.origin_x
        wy = (self.map_data.shape[0] - py - 0.5) * self.resolution + self.origin_y
        return (wx, wy)

    #konverterer en posisjon i meter til en posisjon i bildet.
    def _world_to_pixel(self, wx, wy):
        px = int((wx - self.origin_x) / self.resolution)
        py = int(self.map_data.shape[0] - (wy - self.origin_y) / self.resolution)
        return (px, py)


    #sjekker om en posisjon er åpent.
    def is_free_world(self, wx, wy, inflated=False):
        return self._is_free(*self._world_to_pixel(wx, wy),
                             self.map_data if inflated else self.raw_free)

   #sjekker om et punkt i bildet er åpent.
    def _is_free(self, px, py, mask=None):
        mask = self.map_data if mask is None else mask
        return 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0] and mask[py, px] > 0


     #sjekker om en linje imellom to punkter vil treffe en vegg eller ikke.
    def _check_collision(self, p1, p2, mask=None):
        mask = self.map_data if mask is None else mask
        n = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1])) + 2
        xs = np.rint(np.linspace(p1[0], p2[0], n)).astype(int)
        ys = np.rint(np.linspace(p1[1], p2[1], n)).astype(int)
        if xs.min() < 0 or ys.min() < 0 or xs.max() >= mask.shape[1] or ys.max() >= mask.shape[0]:
            return True
        return bool(np.any(mask[ys, xs] == 0))

    #avstand imellom to punkter i roadmappet.
    def _dist(self, i, j):
        (x1, y1), (x2, y2) = self.nodes[i], self.nodes[j]
        return math.hypot(x2 - x1, y2 - y1)

    #finner næreste ledig punkt dersom en robot er for nærme en vegg.
    def _snap_free(self, px, py, mask, max_r_m=0.6):
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

    #genererer roadmappet i områder robotene kan nå.
    def generate_roadmap(self, start_points_world):
        h, w = self.map_data.shape
        self.region = np.zeros_like(self.map_data)
        for sx, sy in start_points_world:
            seed = self._snap_free(*self._world_to_pixel(sx, sy), self.map_data)
            if seed is None:
                raise RuntimeError(f'Start ({sx:.2f}, {sy:.2f}) is not near free space- '
                                   f'is the robot pose in the map frame')
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

    #legger til punkter i roadmappet og connecter de til deres næreste nabo.
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

    #sjekker om roadmappet er splittet i flere deler.
    def components(self):
        left, comps = set(self.graph), []
        while left:
            c = self._component(next(iter(left)))
            comps.append(c)
            left -= c
        return comps

    #finner alle punkt som kan nås fra et punkt.
    def _component(self, start):
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for v in self.graph[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    #legger til "hindringer" dersom vegger har blitt endret osv. mulig denne ikke vil trenges
    #dersom vi fikk beskjed om at oppgaveteksten var litt utdatert, men vi beholder den som en "safety mesure"
    def add_obstacle(self, wx, wy, radius=0.25):

        px, py = self._world_to_pixel(wx, wy)
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

    #finner nærmeste roadmap til en posisjon.
    def nearest_node(self, wx, wy):
        px, py = self._world_to_pixel(wx, wy)
        usable = [i for i in range(len(self.nodes)) if self.graph[i] and self._is_free(*self.nodes[i])]
        order = sorted(usable or range(len(self.nodes)),
                       key=lambda i: (self.nodes[i][0] - px) ** 2 + (self.nodes[i][1] - py) ** 2)
        for i in order[:40]:
            if not self._check_collision((px, py), self.nodes[i], self.raw_free):
                return i
        return order[0]

    #finner kjøre avstanden fra et punkt til alle andre punkter.
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


    #sjekker om en linje holder en trygg avstand fra andre roboter.
    @staticmethod
    def _seg_clear(p1, p2, avoid_px):
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

    #finner korteste rute imellom to roadmap punkter.
    def astar(self, start, goal, avoid_px=None):
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


    #gjør om et søk til en liste med punkter fra start til slutt.
    @staticmethod
    def _reconstruct(prev, goal):
        path, n = [], goal
        while n is not None:
            path.append(n)
            n = prev[n]
        return path[::-1]

    #gjør at roboten kan kjøre direkte dersom det er en klar vei.
    def _smooth(self, node_path, avoid_px=None):
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

    #planlegger en hel rute imellon to steder.
    def plan_path_world(self, start_xy, goal_xy, avoid=None):
        avoid_px = []
        for ax, ay, r in (avoid or []):
            px, py = self._world_to_pixel(ax, ay)
            avoid_px.append((px, py, r / self.resolution))
        s, g = self.nearest_node(*start_xy), self.nearest_node(*goal_xy)
        path = self.astar(s, g, avoid_px)
        if path is None:
            return []
        pts = self._smooth(path, avoid_px)
        if avoid_px and not self._seg_clear(self._world_to_pixel(*start_xy), pts[0], avoid_px):
            if len(pts) > 1 and self._seg_clear(self._world_to_pixel(*start_xy), pts[1], avoid_px) \
                    and not self._check_collision(self._world_to_pixel(*start_xy), pts[1]):
                pts = pts[1:]
        world = [self._pixel_to_world(*p) for p in pts]
        world.append(tuple(goal_xy))
        return world

    #deler områdene robotene besøker. dette gjør at robotene får sitt "eget" område.
    def _split_goals(self, goals, starts_px):
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

    
    #putter en robots besøks områder i en kort rekkefølge for å ungå at de ikke kjører frem og tilbake.
    def _order_goals(self, start, goals):
        stops = [start] + list(goals)
        sp = {s: self.dijkstra(s) for s in stops} 
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
        while improved:   
            improved = False
            for i in range(1, len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                    if length(cand) < length(tour) - 1e-6:
                        tour, improved = cand, True
        return tour, sp

    
    VIS_RES = 0.05
    CAM_RANGE = 2.2
    CAM_FOV = math.radians(56)
    LOOK_SECTOR = math.radians(50)
    MAX_INCIDENCE = math.radians(60)


    #finner vegger der aruco markers kan henge.
    def _build_visibility(self):
        f = int(round(self.VIS_RES / self.resolution))
        h, w = self.raw_free.shape[0] // f, self.raw_free.shape[1] // f
        occ_frac = cv2.resize((self.raw_free[:h * f, :w * f] == 0).astype(np.float32),
                              (w, h), interpolation=cv2.INTER_AREA)
        self.vis_occ = occ_frac > 0.2
        reach = cv2.resize((self.region[:h * f, :w * f] > 0).astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_AREA) > 0
        k = int(math.ceil((self.robot_radius + 0.1) / self.VIS_RES))
        near = cv2.dilate(reach.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        room = (near > 0) & ~self.vis_occ
        wall = self.vis_occ & (cv2.dilate(room.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        self.wall_id = np.full(self.vis_occ.shape, -1, np.int32)
        wy, wx = np.nonzero(wall)
        self.wall_id[wy, wx] = np.arange(len(wx))
        self.wall_xy = np.stack([(wx + 0.5) * self.VIS_RES + self.origin_x,
                                 (self.vis_occ.shape[0] - wy - 0.5) * self.VIS_RES + self.origin_y], 1)
        r = 3
        offs = np.arange(-r, r + 1, dtype=np.float32)
        kx = np.tile(offs, (2 * r + 1, 1))
        roomf = room.astype(np.float32)
        nx = cv2.filter2D(roomf, -1, kx, borderType=cv2.BORDER_CONSTANT)[wy, wx]
        ny = -cv2.filter2D(roomf, -1, kx.T, borderType=cv2.BORDER_CONSTANT)[wy, wx]
        nn = np.hypot(nx, ny) + 1e-9
        self.wall_normal = np.stack([nx / nn, ny / nn], 1)
        self.seen = np.zeros(len(wx), bool)


    #sender ut "stråler" fra et stedfpr å se hvilke vegger roboten kan se derifra.
    def _cast(self, wx, wy, angles, max_range):
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
        facing = -(np.cos(ra) * self.wall_normal[ids, 0] + np.sin(ra) * self.wall_normal[ids, 1])
        return np.unique(ids[facing > math.cos(self.MAX_INCIDENCE)])

    #markerer alle vegger kameraet ser nåp som "seen"
    def camera_view(self, wx, wy, yaw, n_rays=40):
        a = yaw + np.linspace(-self.CAM_FOV / 2, self.CAM_FOV / 2, n_rays)
        ids = self._cast(wx, wy, a, self.CAM_RANGE)
        self.seen[ids] = True
        return ids

    #velger hvilken vei roboten burde se mot ved et spot.
    def _look_headings(self, wx, wy, ids):
        if len(ids) == 0:
            return []
        p = self.wall_xy[ids]
        ang = np.arctan2(p[:, 1] - wy, p[:, 0] - wx)
        order = np.argsort(ang)
        a, ids = ang[order], ids[order]
        gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
        start = (int(np.argmax(gaps)) + 1) % len(a)
        a = np.unwrap(np.roll(a, -start))
        ids = np.roll(ids, -start)
        looks, i = [], 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[j + 1] - a[i] <= self.LOOK_SECTOR:
                j += 1
            if j - i + 1 >= 3:
                looks.append((float(math.atan2(math.sin((a[i] + a[j]) / 2), math.cos((a[i] + a[j]) / 2))),
                              ids[i:j + 1].copy()))
            i = j + 1
        return looks

    #velger hvor roboten skal stoppe og se rundt seg.
    def _viewpoints(self, reachable, max_candidates=900, min_gain=6):
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

    #lager fulle planen for hvor en robot skal bevege seg og se rudnt.
    def get_exploration_plan(self, robot_starts, robot_index=0):

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

    #samme plan, men bare punktene som skal kjøres igjennom.
    def get_exploration_path(self, robot_starts, robot_index=0):
        return self.get_exploration_plan(robot_starts, robot_index)[0]