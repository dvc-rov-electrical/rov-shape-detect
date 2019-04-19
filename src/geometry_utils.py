import numpy as np

# removes redundant wrapper arrays generated with found contours
def unwrap_contour(cnt):
    return np.array([point[0] for point in cnt])

def angle_vec(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(cos_theta))

def edge_vector_angle(dest1, origin, dest2):
    return angle_vec(dest1 - origin, dest2 - origin)

def angles_of_shape(points):
    angles = []
    for offset in range(len(points)):
        adj1, origin, adj2 = np.roll(points, offset, axis=0)[:3]
        angles += [edge_vector_angle(adj1, origin, adj2)]
    return np.array(angles)

def edge_lengths_of_shape(points):
    lengths = []
    for offset in range(len(points)):
        p1, p2 = np.roll(points, offset, axis=0)[:2]
        lengths += [np.linalg.norm(p1 - p2)]
    return np.array(lengths)

def perfect_circle_contour(x, y, r, num_points):
    perfect_circle = []
    for theta in range(0, num_points):
        perfect_circle += [[[ r * np.sin(theta) + y, r * np.cos(theta) + x ]]]

    perfect_circle = np.array(perfect_circle, np.int32)
    return perfect_circle