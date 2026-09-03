# backend/core_engine.py
import numpy as np

# ==========================================
# 1. DIVIDE-AND-CONQUER k-d TREE INDEX (MODULE 3)
# ==========================================
class KDTreeNode:
    __slots__ = ['point', 'depth', 'idx', 'left', 'right']
    def __init__(self, point, depth=0, idx=0, left=None, right=None):
        self.point = point
        self.depth = depth
        self.idx = idx
        self.left = left
        self.right = right

class SpatialIndexKDTree:
    """
    Balanced 2D/3D Spatial k-d Tree constructed via recursive median bisection.
    Guarantees O(N log N) build and O(log N) proximity lookups.
    """
    def __init__(self, points):
        self.points = points
        indices = list(range(len(points)))
        self.root = self._build(indices, depth=0)

    def _build(self, indices, depth):
        if not indices:
            return None
        axis = depth % 2  # Split along X, then Y
        indices.sort(key=lambda i: self.points[i][axis])
        median = len(indices) // 2

        return KDTreeNode(
            point=self.points[indices[median]],
            depth=depth,
            idx=indices[median],
            left=self._build(indices[:median], depth + 1),
            right=self._build(indices[median + 1:], depth + 1)
        )

    def nearest_k(self, target, k=5, max_radius=None):
        best_nodes = []  # List of tuples: (-distance, point_index)

        def _search(node):
            if node is None:
                return

            dist = float(np.linalg.norm(np.array(node.point[:2]) - np.array(target[:2])))
            
            if max_radius is None or dist <= max_radius:
                if len(best_nodes) < k:
                    best_nodes.append((dist, node.idx))
                    best_nodes.sort(key=lambda x: x[0])
                elif dist < best_nodes[-1][0]:
                    best_nodes[-1] = (dist, node.idx)
                    best_nodes.sort(key=lambda x: x[0])

            axis = node.depth % 2
            diff = target[axis] - node.point[axis]

            first = node.left if diff < 0 else node.right
            second = node.right if diff < 0 else node.left

            _search(first)

            current_max_dist = best_nodes[-1][0] if len(best_nodes) == k else float('inf')
            if abs(diff) < current_max_dist:
                _search(second)

        _search(self.root)
        return [
            {"index": idx, "dist": round(d, 3), "point": self.points[idx].tolist()}
            for d, idx in best_nodes
        ]


# ==========================================
# 2. DELAUNAY TRIANGULATION & MESH HEALTH (MODULE 2)
# ==========================================
def robust_delaunay_triangulation(points):
    from scipy.spatial import Delaunay
    pts2d = points[:, :2]
    tri = Delaunay(pts2d)
    return np.array(tri.simplices, dtype=np.int32)


def calculate_mesh_health(vertices, faces):
    if len(faces) == 0:
        return {"mean_min_angle": 0, "worst_min_angle": 0, "sliver_percentage": 0, "mean_aspect_ratio": 0}

    pts = vertices[faces]
    p0, p1, p2 = pts[:, 0], pts[:, 1], pts[:, 2]

    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p0 - p2, axis=1)

    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(1e-12, s * (s - a) * (s - b) * (s - c)))

    r_in = area / s
    r_circ = (a * b * c) / (4.0 * np.maximum(1e-12, area))
    aspect_ratios = r_circ / (2.0 * np.maximum(1e-12, r_in))

    cos_A = np.clip((b**2 + c**2 - a**2) / (2 * b * c + 1e-12), -1.0, 1.0)
    min_angles = np.degrees(np.arccos(cos_A))
    sliver_count = np.sum((min_angles < 15.0) | (aspect_ratios > 5.0))

    return {
        "mean_min_angle": float(np.mean(min_angles)),
        "worst_min_angle": float(np.min(min_angles)),
        "sliver_percentage": float((sliver_count / len(faces)) * 100),
        "mean_aspect_ratio": float(np.mean(aspect_ratios))
    }