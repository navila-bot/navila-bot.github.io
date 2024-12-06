import cv2
import numpy as np
from typing import List, Tuple
import os
from datetime import timedelta


def get_video_properties(video_path: str) -> Tuple[int, int, int, float]:
    """Get video width, height, frame count, and fps."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, frame_count, fps


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def apply_rounded_corners(image: np.ndarray, radius: int) -> np.ndarray:
    """Apply rounded corners to an image."""
    height, width = image.shape[:2]

    # Create a mask with rounded corners
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, radius), (width - radius, height - radius), 255, -1)

    # Draw the rounded corners
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)

    # Convert mask to 3 channels
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    # Create white background
    background = np.full_like(image, 255)

    # Blend the image with the background using the mask
    rounded_image = image * mask_3d + background * (1 - mask_3d)

    return rounded_image.astype(np.uint8)


def concat_videos_grid(
    video_paths: List[str],
    output_path: str,
    x_padding: int = 10,
    y_padding: int = 10,
    corner_radius: int = 10,
):
    """
    Concatenate multiple videos into a 3-column grid with padding and rounded corners.
    Args:
        video_paths: List of paths to input videos
        output_path: Path for output video
        x_padding: Horizontal padding between videos
        y_padding: Vertical padding between videos
        corner_radius: Radius for rounded corners
    """
    if len(video_paths) == 0:
        raise ValueError("No video paths provided")

    # Open all video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]

    # Get properties of all videos
    properties = [get_video_properties(path) for path in video_paths]
    video_names = [os.path.basename(path) for path in video_paths]

    # Find maximum dimensions among all videos
    max_width = max(prop[0] for prop in properties)
    max_height = max(prop[1] for prop in properties)

    # Calculate grid dimensions
    cols = 3
    rows = (len(video_paths) + cols - 1) // cols  # Ceiling division
    grid_width = (max_width * cols) + (x_padding * (cols - 1))
    grid_height = (max_height * rows) + (y_padding * (rows - 1))

    # Get minimum fps among all videos
    min_fps = min(prop[3] for prop in properties)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, min_fps, (grid_width, grid_height))

    # Track frame positions for each video
    frame_positions = [0 for _ in range(len(caps))]
    frame_counts = [prop[2] for prop in properties]
    fps_values = [prop[3] for prop in properties]

    total_frames_processed = 0
    max_total_frames = sum(frame_counts)

    while True:
        # Create white background
        grid_frame = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)
        should_write_frame = False
        all_completed = True

        # Print overall progress
        progress_percent = (total_frames_processed / max_total_frames) * 100
        print(
            f"\nOverall Progress: {progress_percent:.1f}% ({total_frames_processed}/{max_total_frames} frames)"
        )

        # Print status for each video
        print("\nVideo Status:")
        for idx, (name, pos, count, fps) in enumerate(
            zip(video_names, frame_positions, frame_counts, fps_values)
        ):
            current_time = pos / fps if fps > 0 else 0
            total_time = count / fps if fps > 0 else 0
            status = f"Complete" if pos >= count else f"Processing"
            print(
                f"{name}: {status} - Time: {format_time(current_time)}/{format_time(total_time)} ({(pos/count*100):.1f}%)"
            )

        for idx, cap in enumerate(caps):
            if idx >= len(video_paths):
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

                # Apply rounded corners
                frame = apply_rounded_corners(frame, corner_radius)

                # Calculate position in grid
                x = col * (max_width + x_padding)
                y = row * (max_height + y_padding)

                # Place frame in grid
                grid_frame[y : y + max_height, x : x + max_width] = frame

                # Only increment frame position if we haven't finished first round
                if frame_positions[idx] < frame_counts[idx]:
                    frame_positions[idx] += 1
                    total_frames_processed += 1
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

    # Print final status
    print("\nFinal Status:")
    for idx, (name, count, fps) in enumerate(
        zip(video_names, frame_counts, fps_values)
    ):
        total_time = count / fps if fps > 0 else 0
        print(f"{name}: Complete - Total Duration: {format_time(total_time)}")

    # Release resources
    for cap in caps:
        cap.release()
    out.release()

    print(f"\nVideo concatenation completed successfully!")
    print(f"Output saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    video_paths = [
        "./go2/1_1.mp4",
        "./go2/2_1.mp4",
        "./go2/3_1.mp4",
        "./go2/4_1.mp4",
        "./go2/5_1.mp4",
        "./go2/6_1.mp4",
        "./go2/7_1.mp4",
        "./go2/8_1.mp4",
        "./go2/9_1.mp4",
        "./go2/10_1.mp4",
        "./go2/11_1.mp4",
        "./go2/12_1.mp4",
        "./go2/13_1.mp4",
        "./go2/14_1.mp4",
        "./go2/15_1.mp4",
        # Add more video paths as needed
    ]

    try:
        concat_videos_grid(
            video_paths=video_paths,
            output_path="output_grid.mp4",
            x_padding=10,
            y_padding=10,
            corner_radius=0,
        )
        print("Video concatenation completed successfully!")
    except Exception as e:
        print(f"Error during video concatenation: {str(e)}")
