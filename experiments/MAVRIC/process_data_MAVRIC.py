import sys
import os
import numpy as np
import pandas as pd
import dill
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion
from kalman_filter import NonlinearKinematicBicycle
from sklearn.model_selection import train_test_split
import pdb
nu_path = './devkit/python-sdk/'
sys.path.append(nu_path)
sys.path.append("../../trajectron")
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes
from environment import Environment, Scene, Node, GeometricMap, derivative_of

# scene_blacklist = [499, 515, 517]

FREQUENCY = 2
# dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

curv_0_2 = 0
curv_0_1 = 0
total = 0

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep,
                        non_aug_node=node)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def process_scene(ns_scene_name, env, data_path):
    inst_pq = pd.read_parquet(data_path + '/instance.pq')
    frame_pq = pd.read_parquet(data_path + '/frame.pq')
    scene_id = int(ns_scene_name.split('-')[1])
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])
    inst_scene_pq = inst_pq[inst_pq.scene_ID == scene_id]
    frame_scene_pq = frame_pq[frame_pq.scene_ID == scene_id]
    max_frame_id = int(inst_scene_pq.tail(1).frame_ID)
    for i in range(max_frame_id+1):
        # Vehicle
        data_frame = pd.DataFrame(columns=['frame_id',
                                            'type',
                                            'node_id',
                                            'robot',
                                            'x', 'y', 'z',
                                            'length',
                                            'width',
                                            'height',
                                            'heading'])
        inst_frame_pq = inst_scene_pq[inst_scene_pq.frame_ID == i]
        idx = inst_frame_pq.frame_ID.index
        data_frame.frame_id = inst_frame_pq.frame_ID
        # data_frame.type = inst_frame_pq.obj_type
        category_idx = 0
        bad_idxs = []
        for category in np.array(inst_frame_pq.obj_type):
            if (category == 'pedestrian'):
                data_frame.type.at[idx[category_idx]] = env.NodeType.PEDESTRIAN
            elif (category == 'vehicle'):
                data_frame.type.at[idx[category_idx]] = env.NodeType.VEHICLE
            else:
                bad_idxs.append(category_idx)
            category_idx += 1

        data_frame.node_id = inst_frame_pq.obj_ID.apply(str)
        data_frame.robot = False
        data_frame.x = inst_frame_pq.pos_x
        data_frame.y = inst_frame_pq.pos_y
        data_frame.z = inst_frame_pq.pos_z
        data_frame.length = inst_frame_pq.size_x
        data_frame.width = inst_frame_pq.size_y
        data_frame.height = inst_frame_pq.size_z
        heading_idx = 0
        for arr in np.array(inst_frame_pq.orientation):
            data_frame.heading.at[idx[heading_idx]] = Quaternion(arr).yaw_pitch_roll[0]
            heading_idx += 1
        data_frame = data_frame.drop(idx[bad_idxs])
        data = data.append(data_frame,ignore_index=True)
        # EGO
        data_frame_ego = pd.DataFrame(columns=['frame_id',
                                            'type',
                                            'node_id',
                                            'robot',
                                            'x', 'y', 'z',
                                            'length',
                                            'width',
                                            'height',
                                            'heading',
                                            'orientation'])
        frame_frame_pq = frame_scene_pq[frame_scene_pq.frame_ID == i]
        data_frame_ego.frame_id = frame_frame_pq.frame_ID
        data_frame_ego.type = env.NodeType.VEHICLE
        data_frame_ego.node_id = 'ego'
        data_frame_ego.robot = True
        data_frame_ego.x = frame_frame_pq.ego_pos_x
        data_frame_ego.y = frame_frame_pq.ego_pos_y
        data_frame_ego.z = frame_frame_pq.ego_pos_z
        data_frame_ego.length = 4
        data_frame_ego.width = 1.7
        data_frame_ego.height = 1.5
        data_frame_ego.heading = Quaternion(np.array(frame_frame_pq.ego_orientation)[0]).yaw_pitch_roll[0]
        data_frame_ego.orientation = None
        data = data.append(data_frame_ego,ignore_index=True)
    if len(data.index) == 0:
        return None
    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()

    x_min = np.round(data['x'].min() - 50)
    x_max = np.round(data['x'].max() + 50)
    y_min = np.round(data['y'].min() - 50)
    y_max = np.round(data['y'].max() + 50)

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id), aug_func=augment, x_min=x_min, y_min=y_min)
    # # Generate Maps
    # map_name = nusc.get('log', ns_scene['log_token'])['location']
    # nusc_map = NuScenesMap(dataroot=data_path, map_name=map_name)

    # type_map = dict()
    # x_size = x_max - x_min
    # y_size = y_max - y_min
    # patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
    # patch_angle = 0  # Default orientation where North is up
    # canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
    # homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    # layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
    #                'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
    # map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(
    #     np.uint8)
    # map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
    # # PEDESTRIANS
    # map_mask_pedestrian = np.stack((map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=0)
    # type_map['PEDESTRIAN'] = GeometricMap(data=map_mask_pedestrian, homography=homography, description=', '.join(layer_names))
    # # VEHICLES
    # map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)
    # type_map['VEHICLE'] = GeometricMap(data=map_mask_vehicle, homography=homography, description=', '.join(layer_names))

    # map_mask_plot = np.stack(((np.max(map_mask[:3], axis=0) - (map_mask[3] + 0.5 * map_mask[4]).clip(
    #     max=255)).clip(min=0).astype(np.uint8), map_mask[8], map_mask[9]), axis=0)
    # type_map['VISUALIZATION'] = GeometricMap(data=map_mask_plot, homography=homography, description=', '.join(layer_names))

    # scene.map = type_map
    # del map_mask
    # del map_mask_pedestrian
    # del map_mask_vehicle
    # del map_mask_plot

    for node_id in pd.unique(data['node_id']):
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id]
        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            # print('Occlusion')
            continue  # TODO Make better
        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading'].values
        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE and not node_id == 'ego':
            # Kalman filter Agent
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

            filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
            P_matrix = None
            for i in range(len(x)):
                if i == 0:  # initalize KF
                    # initial P_matrix
                    P_matrix = np.identity(4)
                elif i < len(x):
                    # assign new est values
                    x[i] = x_vec_est_new[0][0]
                    y[i] = x_vec_est_new[1][0]
                    heading[i] = x_vec_est_new[2][0]
                    velocity[i] = x_vec_est_new[3][0]

                if i < len(x) - 1:  # no action on last data
                    # filtering
                    x_vec_est = np.array([[x[i]],
                                          [y[i]],
                                          [heading[i]],
                                          [velocity[i]]])
                    z_new = np.array([[x[i + 1]],
                                      [y[i + 1]],
                                      [heading[i + 1]],
                                      [velocity[i + 1]]])
                    x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                        x_vec_est=x_vec_est,
                        u_vec=np.array([[0.], [0.]]),
                        P_matrix=P_matrix,
                        z_new=z_new
                    )
                    P_matrix = P_matrix_new

            curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
            if pl < 1.0:  # vehicle is "not" moving
                x = x[0].repeat(max_timesteps + 1)
                y = y[0].repeat(max_timesteps + 1)
                heading = heading[0].repeat(max_timesteps + 1)
            global total
            global curv_0_2
            global curv_0_1
            total += 1
            if pl > 1.0:
                if curvature > .2:
                    curv_0_2 += 1
                    node_frequency_multiplier = 3*int(np.floor(total/curv_0_2))
                elif curvature > .1:
                    curv_0_1 += 1
                    node_frequency_multiplier = 3*int(np.floor(total/curv_0_1))

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]
            heading = heading.astype('float64')
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
        # print(node.type)
        node.first_timestep = node_df['frame_id'].iloc[0]
        if node_df.iloc[0]['robot'] == True:
            node.is_robot = True
            scene.robot = node
        scene.nodes.append(node)
    # print(scene)
    return scene


def process_data_MAVRIC(data_path, output_path, val_split):
    global dt;
    if ('NuScene' in data_path):
        dt = 0.5
    elif ('Lyft' in data_path):
        dt = 0.2
    elif ('Argo' in data_path):
        dt = 0.2
    elif ('Waymo' in data_path):
        dt = 0.1
    frame_pq = pd.read_parquet(data_path + '/frame.pq', engine='pyarrow')
    scene_in_frame = np.unique(frame_pq.scene_ID.to_numpy())
    num_of_scenes = len(scene_in_frame)
   
    cnt = 0
    train_scene_names = []
    test_scene_names = []
    val_scene_names = []#val_scenes
    for i in scene_in_frame:
        # if (cnt < int(len(scene_in_frame) * 0.8)):
        train_scene_names.append('scene-' + str(i).zfill(4))
        # else:
            # test_scene_names.append('scene-' + str(i).zfill(4))
        cnt += 1
    ns_scene_names = dict()
    ns_scene_names['train'] = train_scene_names
    ns_scene_names['val'] = val_scene_names
    ns_scene_names['test'] = test_scene_names

    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        env.robot_type = env.NodeType.VEHICLE
        scenes = []
        for ns_scene_name in tqdm(ns_scene_names[data_class]):
            # ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', ns_scene_name)[0])
            # scene_id = int(ns_scene['name'].replace('scene-', ''))
            # if scene_id in scene_blacklist:  # Some scenes have bad localization
                # continue

            scene = process_scene(ns_scene_name, env, data_path)
            if scene is not None:
                if data_class == 'train':
                    scene.augmented = list()
                    angles = np.arange(0, 360, 15)
                    for angle in angles:
                        scene.augmented.append(augment_scene(scene, angle))
                scenes.append(scene)

        print(f'Processed {len(scenes):.2f} scenes')

        env.scenes = scenes

        if len(scenes) > 0:
            dataset_str = str.split(data_path, '/')[-2]
            dataset_num = str.split(data_path, '/')[-1]
            data_dict_path = os.path.join(output_path, dataset_str + '_' + dataset_num +'_full.pkl')
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print('Saved Environment!')

        global total
        global curv_0_2
        global curv_0_1
        print(f"Total Nodes: {total}")
        print(f"Curvature > 0.1 Nodes: {curv_0_1}")
        print(f"Curvature > 0.2 Nodes: {curv_0_2}")
        total = 0
        curv_0_1 = 0
        curv_0_2 = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--val_split', type=int, default=0.15)
    args = parser.parse_args()
    process_data_MAVRIC(args.data, args.output_path, args.val_split)
