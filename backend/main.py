# backend/main.py
import os
import time
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from core_engine import (
    robust_delaunay_triangulation,
    calculate_mesh_health,
    SpatialIndexKDTree
)

app = FastAPI(title="TerrainCraft Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {
    "vertices": None,
    "faces": None,
    "kdtree": None
}

@app.get("/")
def serve_ui():
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))
    return FileResponse(frontend_path)

def process_points_pipeline(points):
    # Deduplication
    _, unique_indices = np.unique(np.round(points[:, :2], 4), axis=0, return_index=True)
    points = points[unique_indices]

    # Module 2: Triangulation
    t0 = time.perf_counter()
    faces = robust_delaunay_triangulation(points)
    dnc_time = (time.perf_counter() - t0) * 1000.0

    # Module 3: Build Spatial k-d Tree
    t_kd0 = time.perf_counter()
    kdtree = SpatialIndexKDTree(points)
    kdtree_build_time = (time.perf_counter() - t_kd0) * 1000.0

    # Module 4: Analytics
    health = calculate_mesh_health(points, faces)

    STATE["vertices"] = points
    STATE["faces"] = faces
    STATE["kdtree"] = kdtree

    return {
        "vertices": points.tolist(),
        "faces": faces.tolist(),
        "metrics": {
            "n_points": len(points),
            "n_faces": len(faces),
            "dnc_latency_ms": round(dnc_time, 2),
            "kdtree_build_ms": round(kdtree_build_time, 2),
            "health": health
        }
    }

@app.get("/api/triangulate/generate")
def triangulate_generated(n_points: int = 1000, complexity: str = "alpine"):
    np.random.seed(42)
    x = np.random.uniform(-80, 80, n_points)
    y = np.random.uniform(-80, 80, n_points)

    if complexity == "alpine":
        z = (np.sin(x / 12.0) * np.cos(y / 12.0) * 25.0 +
             np.sin(x / 4.0 + y / 4.0) * 6.0 +
             np.random.normal(0, 0.4, n_points))
    elif complexity == "faultline":
        z = np.where(x > 0, 18.0 + y * 0.1, -12.0 - y * 0.1) + np.random.normal(0, 0.8, n_points)
    else:
        r = np.sqrt(x**2 + y**2)
        z = 35.0 * np.exp(-0.0008 * (r**2)) - 15.0 * np.exp(-0.005 * (r**2)) + np.random.normal(0, 0.4, n_points)

    points = np.column_stack((x, y, z))
    return process_points_pipeline(points)

@app.post("/api/triangulate/upload")
async def triangulate_upload(file: UploadFile = File(...)):
    """Module 1: Ingests raw CSV or XYZ coordinates [x, y, z]"""
    content = await file.read()
    lines = content.decode("utf-8", errors="ignore").strip().splitlines()
    pts = []
    for line in lines:
        parts = line.replace(",", " ").split()
        if len(parts) >= 3:
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if len(pts) < 4:
        return {"error": "File must contain at least 4 valid 3D points."}
    
    return process_points_pipeline(np.array(pts, dtype=np.float64))

@app.get("/api/nearest")
def find_nearest_neighbors(x: float, y: float, k: int = 5):
    """Module 3: Fast logarithmic nearest neighbor query"""
    if STATE["kdtree"] is None:
        return {"error": "Mesh not built"}
    
    t0 = time.perf_counter()
    neighbors = STATE["kdtree"].nearest_k(target=[x, y], k=k)
    latency_us = (time.perf_counter() - t0) * 1_000_000

    return {
        "query": [x, y],
        "neighbors": neighbors,
        "latency_us": round(latency_us, 1)
    }

@app.get("/api/benchmark")
def run_benchmark_matrix():
    sizes = [100, 250, 500, 1000, 2000, 4000]
    dnc_times = []
    naive_projections = []

    for s in sizes:
        pts = np.random.uniform(-100, 100, (s, 3))
        t0 = time.perf_counter()
        _ = robust_delaunay_triangulation(pts)
        dt = (time.perf_counter() - t0) * 1000.0
        dnc_times.append(round(dt, 2))
        naive_projections.append(round(max(0.1, dnc_times[0]) * ((s / 100) ** 2), 2))

    return {"sizes": sizes, "dnc_ms": dnc_times, "naive_ms": naive_projections}
