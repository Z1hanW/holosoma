import numpy as np
import torch
from scipy import ndimage

from motion_tracking.envs.terrains.subterrain_generator import pyramid_sloped_subterrain, discrete_obstacles_subterrain, \
    pyramid_stairs_subterrain, random_uniform_subterrain, stepping_stones_subterrain, poles_subterrain
from motion_tracking.envs.terrains.terrain_utils import convert_heightfield_to_trimesh
from motion_tracking.envs.terrains.subterrain import SubTerrain


class Terrain:
    def __init__(self, terrain_config, device) -> None:
        self.terrain_config = terrain_config
        self.device = device
        self.num_objects = terrain_config.max_num_objects
        self.spacing_between_objects = terrain_config.spacing_between_objects

        # place objects in the border region
        length = terrain_config.map_length * terrain_config.num_terrains
        objects_per_pass = int(length / self.spacing_between_objects)
        object_rows = 0 if self.num_objects == 0 else int(self.num_objects / objects_per_pass) + 2
        self.object_playground_depth = object_rows * self.spacing_between_objects

        self.horizontal_scale = terrain_config.horizontal_scale
        self.vertical_scale = terrain_config.vertical_scale
        self.border_size = terrain_config.border_size
        self.env_length = terrain_config.map_length
        self.env_width = terrain_config.map_width
        self.proportions = [np.sum(terrain_config.terrain_proportions[:i + 1]) for i in range(len(terrain_config.terrain_proportions))]

        self.env_rows = terrain_config.num_levels
        self.env_cols = terrain_config.num_terrains
        self.num_maps = self.env_rows * self.env_cols

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.object_playground_cols = int(self.object_playground_depth / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border + self.object_playground_cols
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.ceiling_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16) + (3 / self.vertical_scale)

        self.walkable_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.flat_field_raw = np.ones((self.tot_rows, self.tot_cols), dtype=np.int16)

        if self.terrain_config.load_terrain:
            print("Loading a pre-generated terrain")
            params = torch.load(self.terrain_config.terrain_path)
            self.height_field_raw = params["height_field_raw"]
            self.walkable_field_raw = params["walkable_field_raw"]
        else:
            self.generate_subterrains()
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.height_field_raw, self.horizontal_scale, self.vertical_scale, self.terrain_config.slope_threshold
        )
        self.compute_walkable_coords()
        self.compute_flat_coords()

        if self.terrain_config.save_terrain:
            print("Saving this generated terrain")
            torch.save({
                "height_field_raw": self.height_field_raw,
                "walkable_field_raw": self.walkable_field_raw,
                "vertices": self.vertices,
                "triangles": self.triangles,
                "border_size": self.border_size,
            }, self.terrain_config.terrain_path)

    def generate_subterrains(self):
        if self.terrain_config.terrain_composition == "curriculum":
            self.curriculum(n_subterrains_per_level=self.env_cols, n_levels=self.env_rows)
        elif self.terrain_config.terrain_composition == "randomized_subterrains":
            self.randomized_subterrains()
        else:
            raise NotImplementedError("Terrain composition configuration " + self.terrain_config.terrain_composition +
                                      " not implemented")

    def compute_walkable_coords(self):
        self.walkable_field_raw[:self.border, :] = 1
        self.walkable_field_raw[:, -self.border - self.object_playground_cols:] = 1
        self.walkable_field_raw[:, :self.border] = 1
        self.walkable_field_raw[-self.border:, :] = 1

        self.walkable_field = torch.tensor(self.walkable_field_raw, device=self.device)

        walkable_x_indices, walkable_y_indices = torch.where(self.walkable_field == 0)
        self.walkable_x_coords = walkable_x_indices * self.horizontal_scale - self.border_size
        self.walkable_y_coords = walkable_y_indices * self.horizontal_scale - self.border_size

    def compute_flat_coords(self):
        self.flat_field_raw[:self.border, :] = 1
        self.flat_field_raw[:, -self.border - self.object_playground_cols:] = 1
        self.flat_field_raw[:, :self.border] = 1
        self.flat_field_raw[-self.border:, :] = 1

        self.flat_field_raw = torch.tensor(self.flat_field_raw, device=self.device)

        flat_x_indices, flat_y_indices = torch.where(self.flat_field_raw == 0)
        self.flat_x_coords = flat_x_indices * self.horizontal_scale - self.border_size
        self.flat_y_coords = flat_y_indices * self.horizontal_scale - self.border_size

    def sample_valid_locations(self, num_envs):
        x_loc = np.random.randint(0, self.walkable_x_coords.shape[0], size=num_envs)
        y_loc = np.random.randint(0, self.walkable_y_coords.shape[0], size=num_envs)
        valid_locs = torch.stack([self.walkable_x_coords[x_loc], self.walkable_y_coords[y_loc]], dim=-1)
        return valid_locs

    def sample_flat_locations(self, num_envs):
        x_loc = np.random.randint(0, self.flat_x_coords.shape[0], size=num_envs)
        y_loc = np.random.randint(0, self.flat_y_coords.shape[0], size=num_envs)
        flat_locs = torch.stack([self.flat_x_coords[x_loc], self.flat_y_coords[y_loc]], dim=-1)
        return flat_locs

    def randomized_subterrains(self):
        raise NotImplementedError("Randomized subterrains not properly implemented")
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            subterrain = SubTerrain(self.terrain_config, "terrain", device=self.device)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_subterrain(subterrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                    random_uniform_subterrain(subterrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                else:
                    pyramid_sloped_subterrain(subterrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_subterrain(subterrain, step_width=0.31, step_height=step_height, platform_size=3.)
            elif choice < 1.:
                discrete_obstacles_subterrain(subterrain, 0.15, 1., 2., 40, platform_size=3.)

            self.height_field_raw[start_x: end_x, start_y:end_y] = subterrain.height_field_raw

    def curriculum(self, n_subterrains_per_level, n_levels):
        for subterrain_idx in range(n_subterrains_per_level):
            for level_idx in range(n_levels):
                subterrain = SubTerrain(self.terrain_config, "terrain", device=self.device)
                difficulty = level_idx / n_levels
                choice = subterrain_idx / n_subterrains_per_level

                # Heightfield coordinate system
                start_x = self.border + level_idx * self.length_per_env_pixels
                end_x = self.border + (level_idx + 1) * self.length_per_env_pixels
                start_y = self.border + subterrain_idx * self.width_per_env_pixels
                end_y = self.border + (subterrain_idx + 1) * self.width_per_env_pixels

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_subterrain(subterrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_subterrain(subterrain, slope=slope, platform_size=3.)
                    random_uniform_subterrain(subterrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_subterrain(subterrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_subterrain(subterrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                elif choice < self.proportions[5]:
                    stepping_stones_subterrain(subterrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0.,
                                               platform_size=3.)
                elif choice < self.proportions[6]:
                    poles_subterrain(subterrain=subterrain, difficulty=difficulty)
                    self.walkable_field_raw[start_x:end_x, start_y:end_y] = (subterrain.height_field_raw != 0)
                elif choice < self.proportions[7]:
                    subterrain.terrain_name = "flat"

                    flat_border = int(4 / self.horizontal_scale)

                    self.flat_field_raw[start_x + flat_border:end_x - flat_border, start_y + flat_border:end_y - flat_border] = 0
                    # plain walking terrain
                    pass
                self.height_field_raw[start_x: end_x, start_y:end_y] = subterrain.height_field_raw

        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)
