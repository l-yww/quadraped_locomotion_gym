#!/usr/bin/env python3
"""
Reduce binary STL mesh files by random sampling of triangles.
This is a simple decimation that randomly selects a subset of triangles
to keep the mesh under the target triangle count.
"""

import struct
import os
import random

MESH_DIR = "/home/wanghaotian/cowa/quadruped_trot/legged_gym/resources/robots/cowa2/meshes"
BACKUP_DIR = "/home/wanghaotian/cowa/quadruped_trot/legged_gym/resources/robots/cowa2/meshes_original"
TARGET_TRIS = 100000  # target max triangles per mesh

def read_stl(filepath):
    """Read binary STL file, return list of (normal, v1, v2, v3) tuples."""
    with open(filepath, 'rb') as f:
        header = f.read(80)
        num_tri = struct.unpack('<I', f.read(4))[0]
        triangles = []
        for _ in range(num_tri):
            data = struct.unpack('<12fH', f.read(50))
            nx, ny, nz = data[0], data[1], data[2]
            v1 = (data[3], data[4], data[5])
            v2 = (data[6], data[7], data[8])
            v3 = (data[9], data[10], data[11])
            triangles.append(((nx, ny, nz), v1, v2, v3))
    return triangles

def write_stl(filepath, triangles):
    """Write binary STL file."""
    with open(filepath, 'wb') as f:
        f.write(b'\x00' * 80)  # header
        f.write(struct.pack('<I', len(triangles)))
        for (nx, ny, nz), v1, v2, v3 in triangles:
            f.write(struct.pack('<12fH', nx, ny, nz,
                                v1[0], v1[1], v1[2],
                                v2[0], v2[1], v2[2],
                                v3[0], v3[1], v3[2], 0))

def main():
    # Backup original meshes
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"Backing up originals to {BACKUP_DIR}")
        for f in os.listdir(MESH_DIR):
            if f.endswith('.STL'):
                import shutil
                shutil.copy2(os.path.join(MESH_DIR, f), os.path.join(BACKUP_DIR, f))

    stl_files = [f for f in os.listdir(MESH_DIR) if f.endswith('.STL')]
    stl_files.sort()

    total_original = 0
    total_new = 0

    for fname in stl_files:
        filepath = os.path.join(MESH_DIR, fname)
        triangles = read_stl(filepath)
        original_count = len(triangles)
        total_original += original_count

        if original_count <= TARGET_TRIS:
            print(f"{fname}: {original_count} triangles (already <= {TARGET_TRIS}, skipping)")
            total_new += original_count
            continue

        # Random sample to target count
        random.seed(42)  # reproducible
        reduced = random.sample(triangles, TARGET_TRIS)
        write_stl(filepath, reduced)

        file_size = os.path.getsize(filepath)
        size_mb = file_size / (1024 * 1024)
        print(f"{fname}: {original_count} -> {len(reduced)} triangles ({size_mb:.2f} MB)")
        total_new += len(reduced)

    print(f"\nTotal: {total_original} -> {total_new} triangles")
    print("Done! Original meshes backed up in:", BACKUP_DIR)

if __name__ == "__main__":
    main()
