import cv2
import numpy as np
from typing import List, Tuple
import os


def get_video_properties(video_path: str) -> Tuple[int, int, int, float]:
    """Get video width, height, frame count, and fps."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, frame_count, fps


def concat_videos_grid(video_paths: List[str], output_path: str, x_padding: int = 10):
    """
    Concatenate multiple videos into a 5x2 grid with padding.
    Args:
        video_paths: List of paths to input videos
        output_path: Path for output video
        x_padding: Horizontal padding between videos
    """
    if len(video_paths) == 0:
        raise ValueError("No video paths provided")

    # Open all video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]

    # Get properties of all videos
    properties = [get_video_properties(path) for path in video_paths]

    # Find maximum dimensions among all videos
    max_width = max(prop[0] for prop in properties)
    max_height = max(prop[1] for prop in properties)

    # Calculate grid dimensions
    cols, rows = 5, 2
    grid_width = (max_width * cols) + (x_padding * (cols - 1))
    grid_height = (max_height * rows) + (x_padding * (rows - 1))

    # Get minimum fps among all videos
    min_fps = min(prop[3] for prop in properties)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, min_fps, (grid_width, grid_height))

    # Track frame positions for each video
    frame_positions = [0 for _ in range(len(caps))]
    frame_counts = [prop[2] for prop in properties]

    while True:
        # Create white background (255 for all RGB channels)
        grid_frame = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)
        should_write_frame = False
        all_completed = True

        for idx, cap in enumerate(caps):
            if idx >= cols * rows:
                continue

            # Calculate position in grid
            row = idx // cols
            col = idx % cols

            # Read frame
            ret, frame = cap.read()

            # If frame reading failed, reset to beginning
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret:
                # Resize frame to match maximum dimensions
                frame = cv2.resize(frame, (max_width, max_height))

                # Calculate position in grid
                x = col * (max_width + x_padding)
                y = row * (max_height + x_padding)

                # Place frame in grid
                grid_frame[y : y + max_height, x : x + max_width] = frame

                # Only increment frame position if we haven't finished first round
                if frame_positions[idx] < frame_counts[idx]:
                    frame_positions[idx] += 1
                    should_write_frame = True
                    if frame_positions[idx] < frame_counts[idx]:
                        all_completed = False
                # If this video has finished but others haven't, keep showing frames but don't count them
                elif any(
                    pos < count for pos, count in zip(frame_positions, frame_counts)
                ):
                    all_completed = False
                    should_write_frame = True

        # Only write frame if we haven't completed all videos
        if should_write_frame:
            out.write(grid_frame)

        # If all videos completed first round, stop
        if all_completed:
            break

    # Release resources
    for cap in caps:
        cap.release()
    out.release()


# Example usage
if __name__ == "__main__":
    video_paths = [
        "./1.mp4",
        "./2.mp4",
        "./3.mp4",
        "./4.mp4",
        "./5.mp4",
        "./6.mp4",
        "./7.mp4",
        "./8.mp4",
        "./9.mp4",
        "./10.mp4",
        # Add more video paths as needed
    ]

    try:
        concat_videos_grid(
            video_paths=video_paths, output_path="output_grid.mp4", x_padding=10
        )
        print("Video concatenation completed successfully!")
    except Exception as e:
        print(f"Error during video concatenation: {str(e)}")
