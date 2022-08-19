import pickle
import numpy as np

from utils.common.utils import (
    polar_to_cart,
    compute_normals,
    rotation_matrix,
    compute_intersection,
    compute_angle
)

from benchbot_api import ActionResult, Agent, BenchBot


# duplicates code fragment from notebooks/nav/boundary_following.ipynb
def compute_next_pose(observations, first_step=True):
    obs = observations

    scan_points_polar = obs['laser']['scans']
    scan_points_polar = np.array([x for x in scan_points_polar if x[0] < 10])  # remove outliers

    scan_points = polar_to_cart(scan_points_polar)
    scan_points = np.concatenate(
        (
            scan_points,
            np.zeros((scan_points.shape[0], 1))
        ),
        axis=1
    )

    scan_points_rot_trans = scan_points

    z_norm = compute_normals(scan_points_rot_trans)

    fov_x_rad = 1.522150344773997

    shift = 0 if first_step else -fov_x_rad / 3

    phi = np.copy(scan_points_polar[:, 1])
    phi_forward = (phi > shift-fov_x_rad/2) & (phi < shift+fov_x_rad/2)

    delta_phi = (fov_x_rad / 3) / 2
    delta_phi_forward = (phi > shift-delta_phi) & (phi < shift+delta_phi)

    scalar = 4
    avg_norm = scalar * -np.average(z_norm[phi_forward], axis=0)

    theta = fov_x_rad / 3 / 2
    avg_norm_left = rotation_matrix(-theta) @ avg_norm[:2]
    avg_norm_right = rotation_matrix(theta) @ avg_norm[:2]

    indices, = np.where(delta_phi_forward)
    origin_left = scan_points_rot_trans[indices[phi[delta_phi_forward].argmin()]]
    origin_right = scan_points_rot_trans[indices[phi[delta_phi_forward].argmax()]]

    intersection = compute_intersection(origin_left, avg_norm_left, origin_right, avg_norm_right)

    return np.array(intersection), -avg_norm


class MyAgent(Agent):
    def __init__(self):
        self.dest_pickle_filepath = './data/miniroom_1_nav_policy_observations.p'
        self.observations = []
        self.actions_to_take = []
        self.actions_count = 24

    def is_done(self, action_result):
        return (action_result != ActionResult.SUCCESS) or (self.actions_count < 0 and len(self.actions_to_take) == 0)

    def pick_action(self, observations, action_list):
        self.actions_count -= 1
        self.observations.append(observations)

        if len(self.actions_to_take) > 0:
            action = self.actions_to_take.pop()
            return action
        else:
            next_position, next_orientation = compute_next_pose(observations)

            rotation_xy_local = np.array([1, 0])

            next_orientation /= np.linalg.norm(next_orientation)

            move_direction = np.copy(next_position)
            move_distance = np.linalg.norm(move_direction)
            move_direction /= move_distance
            
            if move_distance < 0.1:
                move_angle_1 = compute_angle(next_orientation[:2], rotation_xy_local)
                if not np.isclose(compute_angle(rotation_matrix(move_angle_1) @ rotation_xy_local, next_orientation[:2]), 0):
                    move_angle_1 = -move_angle_1
                step_1 = ('move_angle', {'angle': np.degrees(move_angle_1)})
                
            else:
                move_angle_1 = compute_angle(move_direction, rotation_xy_local)
                if not np.isclose(compute_angle(rotation_matrix(move_angle_1) @ rotation_xy_local, move_direction), 0):
                    move_angle_1 = -move_angle_1

                move_angle_2 = compute_angle(move_direction, next_orientation[:2])
                if not np.isclose(compute_angle(rotation_matrix(move_angle_2) @ move_direction, next_orientation[:2]), 0):
                    move_angle_2 = -move_angle_2

                step_1 = ('move_angle', {'angle': np.degrees(move_angle_1)})
                step_2 = ('move_distance', {'distance': move_distance})
                step_3 = ('move_angle', {'angle': np.degrees(move_angle_2)})
                
                self.actions_to_take.extend([step_3, step_2])
            
            return step_1

    def save_result(self, filename, empty_results, empty_object_fn):
        with open(self.dest_pickle_filepath, 'wb') as f:
            pickle.dump({'observations': self.observations}, f)


if __name__ == '__main__':
    BenchBot(agent=MyAgent()).run()
