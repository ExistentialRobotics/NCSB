import pandas as pd
import numpy as np
import argparse
import torch
import json
import glob
import sys
import os

from pathlib import Path
from tqdm import tqdm
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils_new import inverse_kinematics_analytical
from utils_env import create_obstacles
from main_mppi import run_mppi_control
from robot_cdf import RobotCDF
from robot_sdf import RobotSDF

def parse_args():
    parser = argparse.ArgumentParser(description='Planning Benchmark')
    parser.add_argument('--planner_type', type=str, default='rrt',
                      choices=['rrt', 'bubble'],
                      help='Type of planner to use (default: rrt)')
    parser.add_argument('--num_envs', type=int, default=100,
                      help='Number of environments to test (default: 100)')
    parser.add_argument('--seed', type=int, default=2,
                      help='Random seed (default: 2)')
    parser.add_argument('--early_termination', type=bool, default=True,
                      help='Whether to terminate early when solution is found (default: True)')
    parser.add_argument('--one_goal', type=bool, default=False,
                      help='Whether to use single goal configuration (default: False)')
    return parser.parse_args()


def run_planning_benchmark(planner_type: str = "rrt", num_envs: int = 100, seed: int = 3, early_termination: bool = True, one_goal: bool = False):
    """Run planning benchmark across multiple environments and planners."""
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define goal positions
    # goal_positions = np.array([
    #     [0, 3], [0, 3.5], [-0.5, 3], [-1, 3], [-1.5, 2.5],
    #     [-2, -2], [-2.5, 1.5], [-3, 1],  [-3.5, 0],
    #     [-3.5, -0.5], [-3, -1], [-2.5, -0.5], [-2, -1], [-2, -1.5],
    #     [-0.5, -3], [0, -3]
    # ])

    goal_positions = np.array([
        [0, 3], [0, 3.5], [-3, 1], [-2.5, -0.5]
    ])

    
    
    # Initialize results storage
    results = []
    
    # Initialize robot models
    robot_cdf = RobotCDF(device=device)
    # Initialize SDF model for RRT-based planners
    robot_sdf = RobotSDF(device=device) if planner_type != "bubble" else None
    
    # Setup joint limits
    initial_config = np.array([0., 0.], dtype=np.float32)
    joint_limits = (
        np.full_like(initial_config, -np.pi),  # lower bounds
        np.full_like(initial_config, np.pi)    # upper bounds
    )
    
    # Setup planners based on type
    if planner_type == "bubble":
        from planner.bubble_planner import BubblePlanner
        planners = {
            'bubble': BubblePlanner(
                robot_cdf, joint_limits, max_samples=300, batch_size=5,
                device=device, seed=seed, early_termination=early_termination
            ),
            'bubble_connect': BubblePlanner(
                robot_cdf, joint_limits, max_samples=300, batch_size=5,
                device=device, seed=seed, early_termination=early_termination, planner_type='bubble_connect'
            )
        }
    else:  # rrt
        from planner.rrt_ompl import OMPLRRTPlanner
        planners = {
            'cdf_rrt': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='cdf_rrt'
            ),
            'sdf_rrt': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='sdf_rrt'
            ),
            'lazy_rrt': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='lazy_rrt'
            ),
            'rrt_connect': OMPLRRTPlanner(
                robot_sdf=robot_sdf,
                robot_cdf=robot_cdf,
                robot_fk=None,
                joint_limits=joint_limits,
                device=device,
                seed=seed, 
                planner_type='rrt_connect'
            ),
        }
    
    # Run experiments
    for env_idx in tqdm(range(num_envs), desc="Testing environments"):
        # Create random environment
        rng = np.random.default_rng(env_idx + seed)  
        obstacles = create_obstacles(rng=rng)
        obstacle_points = torch.tensor(np.concatenate(obstacles, axis=0), device=device)
        
        # Randomly select a goal position
        goal_idx = rng.integers(0, len(goal_positions))
        goal_pos = goal_positions[goal_idx]
        print(f"Goal position: {goal_pos}")
        
        # Get goal configurations using IK
        goal_configs = inverse_kinematics_analytical(goal_pos[0], goal_pos[1])
        
        # Convert goal configs to tensor and check CDF values
        goal_configs_tensor = torch.tensor(np.stack(goal_configs), device=device)
        cdf_values = robot_cdf.query_cdf(obstacle_points.unsqueeze(0).expand(len(goal_configs), -1, -1), goal_configs_tensor)

        # Skip trial if any goal configuration has CDF value below threshold
        if torch.any(cdf_values.min(dim=1)[0] < 0.1):
            print(f"Skipping trial - found goal configuration with CDF value below threshold")
            continue

        if one_goal:
            goal_configs = np.array([goal_configs[rng.integers(0, len(goal_configs))]])  # Keep as 2D array
        
        
        # Test planners
        for planner_name, planner in planners.items():
            try:
                # Plan with early termination (single goal case)
                if planner_type == "bubble":
                    result = planner.plan(
                        initial_config, goal_configs, obstacle_points
                    )
                else:
                    result = planner.plan(
                        start_config=initial_config,
                        goal_configs=goal_configs,
                        obstacle_points=obstacle_points,
                        max_time=10.0,
                        early_termination=early_termination
                    )
    
                # Store results
                if result is not None and result['metrics'].success:
                    results.append({
                        'env_idx': env_idx,
                        'planner': planner_name,
                        'num_collision_checks': result['metrics'].num_collision_checks,
                        'path_length': result['metrics'].path_length,
                        'planning_time': result['metrics'].planning_time,
                        'success': True
                    })
                else:
                    results.append({
                        'env_idx': env_idx,
                        'planner': planner_name,
                        'num_collision_checks': float('inf'),
                        'path_length': float('inf'),
                        'planning_time': float('inf'),
                        'success': False
                    })
            
            except Exception as e:
                print(f"Error with planner {planner_name} on env {env_idx}: {str(e)}")
                results.append({
                    'env_idx': env_idx,
                    'planner': planner_name,
                    'num_collision_checks': float('inf'),
                    'path_length': float('inf'),
                    'planning_time': float('inf'),
                    'success': False
                })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Compute statistics
    stats = df.groupby('planner').agg({
        'num_collision_checks': ['mean', 'std'],
        'path_length': ['mean', 'std'],
        'planning_time': ['mean', 'std'],
        'success': 'mean'
    }).round(2)
    
    print(f"\n{planner_type.upper()} Planning Statistics:")
    print(stats)
    
    # Save results (commented out)
    # df.to_csv(f'{planner_type}_planning_results.csv', index=False)
    # stats.to_csv(f'{planner_type}_planning_stats.csv')
    
    return df, stats

###############################################################
#             MPPI BENCHMARK + REPORT FOR 2D
###############################################################
def run_mppi_benchmark_2d(
    num_runs: int = 50,
    seed: int = 5,
    dt: float = 0.02,
    duration: float = 40.0,
    samples: int = 150,
    horizon: int = 20,
    safety_margin: float = 0.2,
    control_bound: float = 2.0,
    bootstrap_iters: int = 50,
    dynamic_obstacles_flag: bool = False,
):
    """
    Run MPPI control on randomly generated 2D environments and save all metrics.
    """
    print(f"\n============================")
    print(f" RUNNING MPPI 2D BENCHMARK  ")
    print(f"============================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device

    np.random.seed(seed)
    torch.manual_seed(seed)

    robot_cdf = RobotCDF(device=device)

    goal_positions = np.array([[0, 3], [0, 3.5], [-3, 1], [-2.5, -0.5]])

    for i in range(num_runs):
        print(f"\n--- MPPI RUN {i+1}/{num_runs} ---")
        # Random environment
        rng = np.random.default_rng(seed + i)
        obstacles = create_obstacles(rng=rng)

        # Random goal
        goal_idx = rng.integers(0, len(goal_positions))
        goal_pos = goal_positions[goal_idx]

        initial_config = np.array([0., 0.], dtype=np.float32)

        tracked, rollouts, safe = run_mppi_control(
            obstacles=obstacles,
            initial_config=initial_config,
            goal_pos=goal_pos,
            robot_cdf=robot_cdf,
            dt=dt,  
            duration=duration,
            samples=samples,
            horizon=horizon,
            safety_margin=safety_margin,
            control_bound=control_bound,
            bootstrap_iters=bootstrap_iters,
            seed=seed + i,
            dynamic_obstacles_flag=dynamic_obstacles_flag,
        )
    print("\nFinished MPPI 2D benchmark.\n")


def load_mppi_2d_metrics(metrics_dir="2Dexamples/results/metrics"):
    """ Load all metrics_mppi_2d_*.json files """
    files = glob.glob(f"{metrics_dir}/metrics_mppi_2d_*.json")

    metrics = []
    for f in sorted(files):
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
                metrics.append(data)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    print(f"Loaded {len(metrics)} MPPI-2D runs.")
    return metrics


def compile_mppi_2d_report(metrics, reports_dir="2Dexamples/results/reports"):
    """ Aggregate MPPI 2D metrics into summary + markdown report """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    if len(metrics) == 0:
        print("No metrics found. Cannot create report.")
        return

    def arr(field):
        return np.array([m[field] for m in metrics])
    
    has_flag = ["dynamic_obstacles" in m for m in metrics]
    if all(has_flag):
        dyn_vals = [bool(m["dynamic_obstacles"]) for m in metrics]
        if all(dyn_vals):
            scenario = "dynamic"
        elif not any(dyn_vals):
            scenario = "static"
        else:
            scenario = "mixed"
    else:
        scenario = "unknown"
    suffix = f"_{scenario}" if scenario is not None else ""

    success_rate = np.mean([m["success"] for m in metrics])

    summary = {
        "num_runs": len(metrics),
        "success_rate": float(success_rate),
        "mean_path_length": float(arr("path_length").mean()),
        "std_path_length": float(arr("path_length").std()),
        "mean_control_time": float(arr("control_time").mean()),
        "std_control_time": float(arr("control_time").std()),
        "mean_bootstrap_time": float(arr("bootstrap_time").mean()),
        "std_bootstrap_time": float(arr("bootstrap_time").std()),
        "mean_collision_checks": float(arr("collision_checks_env").mean()),
        "std_collision_checks": float(arr("collision_checks_env").std()),
        "mean_final_goal_dist": float(arr("final_distance_to_goal").mean()),
        "std_final_goal_dist": float(arr("final_distance_to_goal").std()),
    }

    # Save JSON
    json_path = Path(reports_dir) / f"mppi_2d_benchmark_summary{suffix}.json"
    with open(json_path, "w") as fh:
        json.dump({"summary": summary, "all_runs": metrics}, fh, indent=4)

    # Save Markdown
    md_path = Path(reports_dir) / f"mppi_2d_benchmark_report{suffix}.md"
    with open(md_path, "w") as fh:
        fh.write(f"# MPPI 2D Benchmark Report ({scenario})\n\n")
        fh.write(f"- Number of runs: **{summary['num_runs']}**\n")
        fh.write(f"- Scenario: **{scenario}**\n")
        fh.write(f"- Success rate: **{summary['success_rate']*100:.1f}%**\n\n")
        fh.write("## Performance Statistics\n")
        fh.write(f"- Path length: {summary['mean_path_length']:.3f} ± {summary['std_path_length']:.3f}\n")
        fh.write(f"- Control time: {summary['mean_control_time']:.3f}s ± {summary['std_control_time']:.3f}s\n")
        fh.write(f"- Bootstrap time: {summary['mean_bootstrap_time']:.3f}s ± {summary['std_bootstrap_time']:.3f}s\n")
        fh.write(f"- Collision checks: {summary['mean_collision_checks']:.1f} ± {summary['std_collision_checks']:.1f}\n")
        fh.write(f"- Final goal distance: {summary['mean_final_goal_dist']:.3f} ± {summary['std_final_goal_dist']:.3f}\n")

    print(f"Saved 2D MPPI summary to {json_path}")
    print(f"Saved 2D MPPI Markdown report to {md_path}")

if __name__ == "__main__":
    args = parse_args()
    # Uncomment to run planning benchmark for RRT/Bubble planners
    # df, stats = run_planning_benchmark(planner_type=args.planner_type,num_envs=args.num_envs,seed=args.seed,early_termination=args.early_termination,one_goal=args.one_goal)

    # Uncomment to run MPPI 2D benchmark and generate report
    # run_mppi_benchmark_2d(num_runs=50, seed=1, dynamic_obstacles_flag=True)
    # metrics = load_mppi_2d_metrics()
    # compile_mppi_2d_report(metrics)
