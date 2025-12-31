
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

def merge_geometries(urdf_paths, output_path):
    # Initialize a new robot element
    new_root = ET.Element("robot", name="merged_object")
    new_link = ET.SubElement(new_root, "link", name="object")
    
    # Iterate over each URDF path
    for urdf_path in urdf_paths:
        # Parse the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Find the 'link' element (assuming there is only one and it's named 'object')
        link_element = root.find(".//link[@name='object']")
        
        # If a link is found, proceed to extract and append visual and collision elements
        if link_element is not None:
            # Append visual and collision elements to the new link element
            for visual in link_element.findall("visual"):
                new_link.append(visual)
            for collision in link_element.findall("collision"):
                new_link.append(collision)

    # Convert the new root element into a tree and write to the specified output file
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding='unicode', xml_declaration=True)


def compute_box_corners(size, origin):
    x, y, z = map(float, size.split())
    ox, oy, oz = map(float, origin['xyz'].split())
    rpy = list(map(float, origin['rpy'].split())) if 'rpy' in origin else (0.0, 0.0, 0.0)
    
    # Create rotation matrix from RPY (roll, pitch, yaw)
    R = RPY_to_rotation_matrix(*rpy)
    
    # Half sizes for corner calculation
    dx, dy, dz = x / 2, y / 2, z / 2
    corners = np.array([
        [dx, dy, dz],
        [dx, dy, -dz],
        [dx, -dy, dz],
        [dx, -dy, -dz],
        [-dx, dy, dz],
        [-dx, dy, -dz],
        [-dx, -dy, dz],
        [-dx, -dy, -dz]
    ])
    
    # Rotate and translate corners
    corners = np.dot(R, corners.T).T + [ox, oy, oz]
    face_centers = calculate_face_centers(corners)

    return corners, face_centers

def calculate_face_centers(corners):
    face_indices = [
        [0, 1, 3, 2],  # Front face
        [4, 5, 7, 6],  # Back face
        [0, 1, 5, 4],  # Top face
        [2, 3, 7, 6],  # Bottom face
        [0, 2, 6, 4],  # Left face
        [1, 3, 7, 5]   # Right face
    ]
    face_centers = []
    for indices in face_indices:
        face = corners[indices]
        center = face.mean(axis=0)
        face_centers.append(center)
    return np.array(face_centers)

def RPY_to_rotation_matrix(roll, pitch, yaw):
    # Compute rotation matrix from roll, pitch, yaw
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [   -sp,          cp*sr,          cp*cr]
    ])
    return R

def read_urdf(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []
    all_corners = []
    all_face_points = []
    for visual in root.findall(".//visual"):
        geometry = visual.find('geometry')
        box = geometry.find('box')
        if box is not None:
            size = box.get('size')
            origin = visual.find('origin').attrib
            corners,face_points = compute_box_corners(size, origin)
            boxes.append({'size': size, 'origin': origin, 'corners': corners})
            all_corners.append(corners)
            all_face_points.append(face_points)
    
    return boxes, all_corners, all_face_points

def create_urdf_spheres_from_points(points, radius=0.05):
    """ Generate URDF XML content for spheres at given points.

    Args:
    points (np.array): Array of points where each row is [x, y, z].
    radius (float): Radius of each sphere.

    Returns:
    str: A string containing URDF XML content.
    """
    urdf_content = []
    for point in points:
        x, y, z = point
        sphere_visual = f"""
        <visual>
          <origin xyz="{x:.8f} {y:.8f} {z:.8f}"/>
          <geometry>
            <sphere radius="{radius}"/>
          </geometry>
        </visual>
        """
        urdf_content.append(sphere_visual)
    
    return "\n".join(urdf_content)

# Example usage
def save_points(filename, save_name, save=True):
    boxes, all_corners, all_face_points = read_urdf(filename)
    all_corners = np.vstack(all_corners)
    all_face_points = np.vstack(all_face_points)

    points = np.concatenate([all_face_points,all_corners],axis=0).astype(np.float32) # Example: generate 100 random 3D points
    
    if save:
        np.save(save_name,points)
    
    return points

def add_noise_to_values(values, noise_level, rpy=True):
    """ Adds Gaussian noise to the given array of values. """
    noise = np.random.normal(0, noise_level, values.shape)
    if rpy:
        values = values + noise * (1,1,1) 
    else:
        values = values + noise
    return values  

def scale_values(values, scale_std):
    """ Scale the values by a random factor with mean=1 and a specified std deviation. """
    scale_factor = np.random.normal(1, scale_std, values.shape)
    return values * scale_factor

def parse_xyz_rpy_size(text):
    """ Parse a space-separated string into a float array. """
    return np.array(list(map(float, text.split())))

def stringify_xyz_rpy_size(values):
    """ Convert an array of values back to a space-separated string. """
    return ' '.join(f"{val:.4f}" for val in values)

def add_noise_to_urdf(file_path, noise_levels, size_scale_std, save_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define the level of noise for position, rotation, and the std deviation for size scaling
    pos_noise, rpy_noise = noise_levels

    # Process each 'link' element to handle both visual and collision consistently
    for link in root.findall('.//link'):
        # Pair each visual with its corresponding collision based on ordering
        pairs = zip(link.findall('.//visual'), link.findall('.//collision'))
        
        # Only modify the first two pairs
        for index, (visual, collision) in enumerate(pairs):
            # if index >= 2:  # Stop after modifying the first two pairs
            #     break
            
            visual_origin = visual.find('.//origin')
            collision_origin = collision.find('.//origin')
            visual_box = visual.find('.//geometry/box')
            collision_box = collision.find('.//geometry/box')
            
            if visual_origin is not None and visual_box is not None:
                xyz = parse_xyz_rpy_size(visual_origin.get('xyz'))
                rpy = parse_xyz_rpy_size(visual_origin.get('rpy'))
                size = parse_xyz_rpy_size(visual_box.get('size'))

                xyz_noised = add_noise_to_values(xyz, pos_noise, False)
                rpy_noised = add_noise_to_values(rpy, rpy_noise, True)
                size_scaled = scale_values(size, size_scale_std)

                visual_origin.set('xyz', stringify_xyz_rpy_size(xyz_noised))
                visual_origin.set('rpy', stringify_xyz_rpy_size(rpy_noised))
                visual_box.set('size', stringify_xyz_rpy_size(size_scaled))

                if collision_origin is not None and collision_box is not None:
                    # Apply the same noise and scaling to collision
                    collision_origin.set('xyz', stringify_xyz_rpy_size(xyz_noised))
                    collision_origin.set('rpy', stringify_xyz_rpy_size(rpy_noised))
                    collision_box.set('size', stringify_xyz_rpy_size(size_scaled))

    # Save the modified URDF
    tree.write(save_path)