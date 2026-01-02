import numpy as np
import torch


class DummyTerrain:
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
        self.flat_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.heightsamples = self.height_field_raw
        self.compute_walkable_coords()
        self.compute_flat_coords()

    def generate_subterrains(self):
        raise NotImplementedError("Dummy terrain does not create any real terrains!")

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
