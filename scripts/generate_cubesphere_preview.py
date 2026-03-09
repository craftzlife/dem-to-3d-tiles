"""Generate a cube-sphere OBJ model for visualization.

Creates a unit sphere (radius=1) with 6 faces as separate groups,
each with a visible grid so you can see the quadtree structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from src.cube_sphere import face_uv_to_xyz
from src.config import FACE_NAMES

RESOLUTION = 32  # vertices per face edge
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "cubesphere_preview.obj")


def generate():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    vertex_offset = 0

    with open(OUTPUT, "w") as f:
        f.write("# Cube-Sphere Preview Model\n")
        f.write("# 6 faces, each as a separate group\n")
        f.write(f"# Resolution: {RESOLUTION}x{RESOLUTION} per face\n\n")

        for face in range(6):
            f.write(f"\n# === {FACE_NAMES[face]} ===\n")
            f.write(f"g {FACE_NAMES[face]}\n")

            # Shrink UV range slightly (0.01 to 0.99) to create gaps between faces
            margin = 0.008
            u_vals = [margin + (1.0 - 2 * margin) * i / (RESOLUTION - 1) for i in range(RESOLUTION)]
            v_vals = [margin + (1.0 - 2 * margin) * i / (RESOLUTION - 1) for i in range(RESOLUTION)]

            # Write vertices
            for v in v_vals:
                for u in u_vals:
                    x, y, z = face_uv_to_xyz(face, u, v)
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

            # Write normals (same as vertex direction for a sphere)
            for v in v_vals:
                for u in u_vals:
                    x, y, z = face_uv_to_xyz(face, u, v)
                    f.write(f"vn {x:.6f} {y:.6f} {z:.6f}\n")

            # Write texture coordinates
            for v in v_vals:
                for u in u_vals:
                    f.write(f"vt {u:.6f} {v:.6f}\n")

            # Write faces (triangles)
            for r in range(RESOLUTION - 1):
                for c in range(RESOLUTION - 1):
                    tl = vertex_offset + r * RESOLUTION + c + 1       # OBJ is 1-indexed
                    tr = vertex_offset + r * RESOLUTION + (c + 1) + 1
                    bl = vertex_offset + (r + 1) * RESOLUTION + c + 1
                    br = vertex_offset + (r + 1) * RESOLUTION + (c + 1) + 1
                    f.write(f"f {tl}/{tl}/{tl} {bl}/{bl}/{bl} {tr}/{tr}/{tr}\n")
                    f.write(f"f {tr}/{tr}/{tr} {bl}/{bl}/{bl} {br}/{br}/{br}\n")

            vertex_offset += RESOLUTION * RESOLUTION

    verts_total = 6 * RESOLUTION * RESOLUTION
    tris_total = 6 * (RESOLUTION - 1) * (RESOLUTION - 1) * 2
    print(f"Written: {OUTPUT}")
    print(f"  6 faces, {verts_total} vertices, {tris_total} triangles")
    print(f"  Each face is a separate group — visible as seams in 3D viewers")


if __name__ == "__main__":
    generate()
