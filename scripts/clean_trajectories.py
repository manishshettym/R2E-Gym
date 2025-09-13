import json
from pathlib import Path
import sys
sys.path.append('/home/gcpuser/R2E-Gym/src')

from r2egym.agenthub.trajectory import Trajectory

def remove_empty_trajectories(jsonl_file_path):
    """
    Load trajectories, print docker_image fields, then remove lines with empty output_patch.
    """
    jsonl_file = Path(jsonl_file_path)
    
    if not jsonl_file.exists():
        print(f"File {jsonl_file_path} does not exist!")
        return
    
    print(f"Processing file: {jsonl_file_path}")
    print("=" * 50)
    
    # First pass: read and print docker_image fields
    print("Docker images in the file:")
    docker_images = []
    valid_trajectories = []
    
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                trajectory = Trajectory.load_from_model_dump_json(line)
                docker_image = trajectory.docker_image
                output_patch = trajectory.true_output_patch
                
                docker_images.append(docker_image)
                print(f"Line {line_num}: {docker_image}")
                
                # Keep trajectory if output_patch is not empty
                if output_patch.strip():
                    valid_trajectories.append(line)
                else:
                    print(f"  -> Will remove (empty output_patch)")
                    
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print("=" * 50)
    print(f"Total trajectories: {len(docker_images)}")
    print(f"Trajectories with non-empty output_patch: {len(valid_trajectories)}")
    print(f"Trajectories to remove: {len(docker_images) - len(valid_trajectories)}")
    
    # # Ask for confirmation before deleting
    if len(valid_trajectories) < len(docker_images):
        response = input("\nDo you want to proceed with removing trajectories with empty output_patch? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Second pass: write back only valid trajectories
    backup_file = jsonl_file.with_suffix('.jsonl.backup')
    print(f"\nCreating backup: {backup_file}")
    jsonl_file.rename(backup_file)
    
    print(f"Writing cleaned file: {jsonl_file}")
    with open(jsonl_file, 'w') as f:
        for trajectory_line in valid_trajectories:
            f.write(trajectory_line + '\n')
    
    print("Done!")
    print(f"Original file backed up as: {backup_file}")
    print(f"Cleaned file saved as: {jsonl_file}")

if __name__ == "__main__":
    # Process the specified file
    jsonl_file_path = "/home/gcpuser/R2E-Gym/traj/gpt-5-swebv-eval-fusion_k_8.jsonl"
    remove_empty_trajectories(jsonl_file_path)
