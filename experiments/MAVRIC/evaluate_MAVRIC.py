import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
import pdb
sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from scipy.interpolate import RectBivariateSpline

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
args = parser.parse_args()


def compute_road_violations(predicted_trajs, map, channel):
    obs_map = 1 - map.data[..., channel, :, :] / 255

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                         range(obs_map.shape[1]),
                                         obs_map,
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1], old_shape[2]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)

    return num_viol_trajs


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            ############### MOST LIKELY Z ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            # durations = np.array([])
            durations = []
            # num_nodes = np.array([])
            num_nodes = []
            # names = np.array([])
            names = []
            descriptions = []
            print("-- Evaluating GMM Z Mode (Most Likely)")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)

                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_future_timesteps=8,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)  # This will trigger grid sampling

                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=False,
                                                                       kde=False)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                for i in range(len(batch_error_dict[args.node_type]['ade'])):
                    # durations = np.hstack((durations, scene.duration()))
                    # num_nodes = np.hstack((num_nodes, len(scene.nodes)))
                    # names = np.hstack((names, scene.name))
                    # descriptions.append(scene.description)
                    durations.append(scene.duration())
                    num_nodes.append(len(scene.nodes))
                    names.append(scene.name)
                    descriptions.append(scene.description)

            print(np.mean(eval_fde_batch_errors))
            pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml', 
                          'duration':np.array(durations), 'num nodes':np.array(num_nodes), 'scene names':np.array(names), 'descriptions':descriptions}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_ade_most_likely_z.csv'))
            pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml',
                          'duration':durations, 'num nodes':num_nodes, 'scene names':names, 'descriptions':descriptions}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_fde_most_likely_z.csv'))


            ############### FULL ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            
            durations = []
            num_nodes = []
            names = []
            descriptions = []
            
            durations_kde = []
            num_nodes_kde = []
            names_kde = []
            descriptions_kde = []
            
            print("-- Evaluating Full")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=20, #2000
                                               min_future_timesteps=8,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)

                if not predictions:
                    continue

                prediction_dict, _, _ = utils.prediction_output_to_trajectories(predictions,
                                                                                scene.dt,
                                                                                max_hl,
                                                                                ph,
                                                                                prune_ph_to_future=False)
                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=False)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))
                for i in range(len(batch_error_dict[args.node_type]['ade'])):
                    durations.append(scene.duration())
                    num_nodes.append(len(scene.nodes))
                    names.append(scene.name)
                    descriptions.append(scene.description)
                for i in range(len(batch_error_dict[args.node_type]['kde'])):
                    durations_kde.append(scene.duration())
                    num_nodes_kde.append(len(scene.nodes))
                    names_kde.append(scene.name)
                    descriptions_kde.append(scene.description)
                    
        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full',
                      'duration':np.array(durations), 'num nodes':np.array(num_nodes), 'scene names':np.array(names), 'descriptions':descriptions}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_ade_full.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full',
                      'duration':np.array(durations), 'num nodes':np.array(num_nodes), 'scene names':np.array(names), 'descriptions':descriptions}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_fde_full.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full',
                      'duration':np.array(durations_kde), 'num nodes':np.array(num_nodes_kde), 'scene names':np.array(names_kde), 'descriptions':descriptions_kde}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_kde_full.csv'))
        # pd.DataFrame({'value': eval_road_viols, 'metric': 'road_viols', 'type': 'full',
        #               'duration':durations, 'num nodes':num_nodes, 'scene names':names, 'descriptions':descriptions}
        #              ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_rv_full.csv'))
