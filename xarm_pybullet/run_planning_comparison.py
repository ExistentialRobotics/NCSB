import torch
import numpy as np
from xarm_planning import XArmSDFVisualizer
from planner.rrt_ompl import OMPLRRTPlanner
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

def run_planning_comparison(num_trials=50, random_seed=42, save_dir='results'):
    """Run systematic comparison of different planning methods
    
    Args:
        num_trials: Number of successful trials to collect
        random_seed: Random seed for reproducibility
        save_dir: Directory to save results
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Initialize visualizer with dummy goal (will be updated)
    dummy_goal = torch.tensor([0.0, 0.0, 0.5], device='cuda')
    visualizer = XArmSDFVisualizer(dummy_goal, use_gui=False)
    
    # Define planners to test
    planners = {
        'bubble_cdf': 'Bubble-CDF',
        'rrt': 'Custom RRT',
        'rrt_connect': 'Custom RRT-Connect',
        'ompl_rrt': 'OMPL RRT',
        'ompl_rrt_connect': 'OMPL RRT-Connect'
    }
    
    # Store all results
    results = {name: [] for name in planners.keys()}
    results['test_settings'] = {
        'num_trials': num_trials,
        'random_seed': random_seed,
        'timestamp': timestamp
    }
    
    # Initialize OMPL planners
    ompl_planners = {
        'ompl_rrt': OMPLRRTPlanner(
            robot_sdf=visualizer.robot_sdf,
            robot_fk=visualizer.robot_fk,
            joint_limits=(
                visualizer.robot_fk.joint_limits[:, 0].cpu().numpy(),
                visualizer.robot_fk.joint_limits[:, 1].cpu().numpy()
            ),
            planner_type='rrt'
        ),
        'ompl_rrt_connect': OMPLRRTPlanner(
            robot_sdf=visualizer.robot_sdf,
            robot_fk=visualizer.robot_fk,
            joint_limits=(
                visualizer.robot_fk.joint_limits[:, 0].cpu().numpy(),
                visualizer.robot_fk.joint_limits[:, 1].cpu().numpy()
            ),
            planner_type='rrt_connect'
        )
    }
    
    completed_trials = 0
    attempts = 0
    max_attempts = num_trials * 3
    
    while completed_trials < num_trials and attempts < max_attempts:
        attempts += 1
        
        # Generate random goal in task space
        goal = torch.tensor([
            np.random.uniform(-0.5, 0.5),  # x
            np.random.uniform(-0.5, 0.5),  # y
            np.random.uniform(0.0, 1.0)    # z
        ], device='cuda')
        
        print(f"\nTrial {completed_trials + 1}/{num_trials} (Attempt {attempts})")
        print(f"Testing goal position: {goal.cpu().numpy()}")
        
        # Find goal configuration
        goal_config_with_metrics = visualizer._find_goal_configuration(goal)
        if goal_config_with_metrics is None:
            print("Invalid goal - retrying with new random goal")
            continue
        # Extract just configs (backward compatible)
        goal_config = [config for config, _, _, _ in goal_config_with_metrics]
        
        trial_success = True
        
        # Test each planner
        for planner_name, planner_display_name in planners.items():
            print(f"\nTesting {planner_display_name}")
            
            # Reset robot to initial configuration
            initial_config = torch.zeros(6, device='cuda')
            visualizer.set_robot_configuration(initial_config)
            
            # Run appropriate planner
            if planner_name.startswith('ompl'):
                result = ompl_planners[planner_name].plan(
                    start_config=initial_config.cpu().numpy(),
                    goal_config=goal_config,
                    obstacle_points=visualizer.points_robot
                )
                metrics = result['metrics']
            else:
                # Use visualizer's built-in planners
                visualizer.planner_type = planner_name
                metrics = visualizer.run_demo(execute_trajectory=False)
            
            if metrics and metrics.success:
                # Store metrics and goal information
                results[planner_name].append({
                    'goal_position': goal.cpu().numpy().tolist(),
                    'num_collision_checks': metrics.num_collision_checks,
                    'path_length': float(metrics.path_length),
                    'num_samples': metrics.num_samples,
                    'planning_time': float(metrics.planning_time)
                })
                print(f"Planning succeeded:")
                print(f"Collision checks: {metrics.num_collision_checks}")
                print(f"Path length: {metrics.path_length:.3f}")
                print(f"Number of samples: {metrics.num_samples}")
                print(f"Planning time: {metrics.planning_time:.3f}s")
            else:
                print(f"Planning failed for {planner_display_name}")
                trial_success = False
        
        if trial_success:
            completed_trials += 1
            
            # Save intermediate results
            save_results(results, save_dir, timestamp)
    
    # Print and save final statistics
    print_statistics(results, planners)
    save_results(results, save_dir, timestamp, final=True)
    
def save_results(results, save_dir, timestamp, final=False):
    """Save results to JSON and CSV files"""
    prefix = 'final' if final else 'intermediate'
    
    # Save raw results as JSON
    json_path = save_dir / f'{prefix}_planning_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create pandas DataFrame for easy analysis
    df_data = []
    for planner_name, trials in results.items():
        if planner_name != 'test_settings':
            for trial in trials:
                trial_data = trial.copy()
                trial_data['planner'] = planner_name
                df_data.append(trial_data)
    
    if df_data:
        df = pd.DataFrame(df_data)
        csv_path = save_dir / f'{prefix}_planning_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)

def print_statistics(results, planners):
    """Print summary statistics for all planners"""
    print("\nFinal Statistics:")
    print("-" * 50)
    
    # Create summary table
    headers = ['Planner', 'Success Rate', 'Avg Checks', 'Avg Length', 'Avg Samples', 'Avg Time']
    rows = []
    
    for planner_name, display_name in planners.items():
        trials = results[planner_name]
        if trials:
            num_trials = len(trials)
            success_rate = num_trials / results['test_settings']['num_trials']
            avg_checks = np.mean([t['num_collision_checks'] for t in trials])
            avg_length = np.mean([t['path_length'] for t in trials])
            avg_samples = np.mean([t['num_samples'] for t in trials])
            avg_time = np.mean([t['planning_time'] for t in trials])
            
            rows.append([
                display_name,
                f"{success_rate:.1%}",
                f"{avg_checks:.1f}",
                f"{avg_length:.3f}",
                f"{avg_samples:.1f}",
                f"{avg_time:.3f}s"
            ])
    
    # Print table
    row_format = "{:>20}" * len(headers)
    print(row_format.format(*headers))
    print("-" * (20 * len(headers)))
    for row in rows:
        print(row_format.format(*row))

if __name__ == "__main__":
    run_planning_comparison(num_trials=50, random_seed=42) 