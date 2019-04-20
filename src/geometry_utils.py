import numpy as np

# removes redundant wrapper arrays generated with found contours
def unwrap_contour(cnt):
    return np.array([point[0] for point in cnt])

# Calculates the distance between two points
def distance(s1, s2):
    return np.linalg.norm(s1 - s2)

# Calculates the angle between 2 vectors
def angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(cos_theta))

# Calculates the angle between 2 edges of a polygon
def edge_angle(dest1, origin, dest2):
    return angle(dest1 - origin, dest2 - origin)

# Calculates all angles of a polygon, given its vertice locations
def angles_of_shape(points):
    angles = []
    for offset in range(len(points)):
        adj1, origin, adj2 = np.roll(points, offset, axis=0)[:3]
        angles += [edge_angle(adj1, origin, adj2)]
    return np.array(angles)

# Calculates all edge lengths of a polygon, given its vertice locations
def edges_of_shape(points):
    lengths = []
    for offset in range(len(points)):
        p1, p2 = np.roll(points, offset, axis=0)[:2]
        lengths += [np.linalg.norm(p1 - p2)]
    return np.array(lengths)

# Create a contour that represents a perfect circle (used for circle detection)
def perfect_circle_contour(x, y, r, num_points):
    perfect_circle = []
    for theta in range(0, num_points):
        perfect_circle += [[[ r * np.sin(theta) + y, r * np.cos(theta) + x ]]]

    perfect_circle = np.array(perfect_circle, np.int32)
    return perfect_circle