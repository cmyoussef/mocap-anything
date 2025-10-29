from pathlib import Path


def get_image_pattern_and_frame_range(image_path):
    """
    Given an image path, determine whether it's part of an image sequence.
    If it is, return the sequence pattern (with frame padding) and the (start_frame, end_frame).
    Otherwise, return the original path and None.

    Unlike the previous approach of determining padding from the largest frame,
    this version *preserves* the zero-padding found in the input file's last numeric part.

    Example:
        If image_path is "init_img.0120.png",
        this function might return:
            (
                "E:/.../init_img.%04d.png",  # Preserves 4-digit padding
                (2, 131)
            )
        even if the largest frame is only 3 digits (like 131).
    """
    file_path = Path(image_path)

    # Decide which splitter to use
    # Use '.' if the stem contains a dot, otherwise '_'
    splitter = '.' if file_path.stem.count('.') > 0 else '_'

    # Split the filename stem
    name_parts = file_path.stem.split(splitter)
    parent_dir = file_path.parent

    # If the last part is purely numeric, treat it as a frame number
    if len(name_parts) > 1 and name_parts[-1].isdigit():
        base_name = splitter.join(name_parts[:-1])
        # Original zero-padding from the last part of THIS file
        # E.g., "0120" -> 4 digits
        padding = len(name_parts[-1])

        extension_no_dot = file_path.suffix.lstrip('.')
        # Build a glob pattern to find all matching frames
        glob_pattern = f"{base_name}{splitter}*.{extension_no_dot}"

        seq_files = sorted(parent_dir.glob(glob_pattern))

        if seq_files:
            frame_numbers = []
            for seq_f in seq_files:
                seq_name_parts = seq_f.stem.split(splitter)
                if seq_name_parts and seq_name_parts[-1].isdigit():
                    frame_numbers.append(int(seq_name_parts[-1]))

            if frame_numbers:
                start_frame = min(frame_numbers)
                end_frame = max(frame_numbers)

                # Use the preserved padding from the input file
                sequence_pattern = f"{parent_dir.as_posix()}/{base_name}{splitter}%0{padding}d.{extension_no_dot}"

                return sequence_pattern, (start_frame, end_frame)

    # If not recognized as a sequence, return the original path and None
    return file_path.as_posix(), None


# Example usage:
if __name__ == "__main__":
    image_file = r"/mocap-anything/assets/deer/init_img.0120.png"
    pattern, frame_range = get_image_pattern_and_frame_range(image_file)
    print("Pattern:", pattern)
    print("Frame Range:", frame_range)
