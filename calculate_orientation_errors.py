"""Calculate positioning errors caused by drone attitude errors.

For each configured height and pitch, the ray from the image center is
intersected with the ENU plane z=0. The reference position is compared with
the positions calculated after adding both +0.1 and -0.1 degree to different
combinations of yaw, pitch, and roll.

Examples:
    python calculate_orientation_errors.py --resolution 1920 1080
    python calculate_orientation_errors.py --resolution 1920 1080 --k tests/UFRJ/K-HD.json
    python calculate_orientation_errors.py --resolution 1920 1080 --output results/errors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils.geometry import Geometry


HEIGHTS_M = (90.0, 100.0, 110.0, 120.0)
PITCHES_DEGREES = tuple(float(pitch) for pitch in range(-10, -91, -10))
ANGULAR_ERROR_VALUES_DEGREES = (0.1, -0.1)

# Flags indicate which of (yaw, pitch, roll) receive the angular error.
PERTURBATIONS = {
    "yaw": (True, False, False),
    "pitch": (False, True, False),
    "roll": (False, False, True),
    "yaw+pitch": (True, True, False),
    "yaw+roll": (True, False, True),
    "pitch+roll": (False, True, True),
    "yaw+pitch+roll": (True, True, True),
}


def load_intrinsic_matrix(path: Path) -> np.ndarray:
    """Load and validate a 3 x 3 intrinsic matrix from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        intrinsic_matrix = np.asarray(json.load(file), dtype=float)

    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            f"K must have dimensions 3 x 3; received: {intrinsic_matrix.shape}"
        )
    if not np.all(np.isfinite(intrinsic_matrix)):
        raise ValueError("K contains non-finite values.")
    if intrinsic_matrix[0, 0] == 0.0 or intrinsic_matrix[1, 1] == 0.0:
        raise ValueError("The fx and fy focal lengths in K must be nonzero.")

    return intrinsic_matrix


def get_default_k_path() -> Path:
    """Get K_path from parameters.json while keeping the script configurable."""
    parameters_path = Path("parameters.json")
    if not parameters_path.is_file():
        raise FileNotFoundError(
            "parameters.json was not found. Provide the matrix with --k PATH."
        )

    with parameters_path.open("r", encoding="utf-8") as file:
        parameters = json.load(file)

    k_path = parameters.get("K_path")
    if not k_path:
        raise ValueError("K_path is not defined in parameters.json; use --k PATH.")
    return Path(k_path)


def get_intersection_from_click(
    click: tuple[float, float],
    intrinsic_matrix: np.ndarray,
    drone_rotation: np.ndarray,
    drone_world_position: np.ndarray,
) -> np.ndarray:
    """Replicate the flat-ground intersection required from Geodetic.

    The coordinate system is the same as in
    Geodetic.get_intersection_from_click: East, North, Up (ENU). The ground is
    represented by the plane Up = 0.
    """
    inverse_k = Geometry.inv_K(intrinsic_matrix)
    camera_world_rotation = (
        Geometry.droneToMundoR @ drone_rotation @ Geometry.cameraToDroneR
    )
    homogeneous_pixel = np.array([[click[0]], [click[1]], [1.0]])
    direction = camera_world_rotation @ inverse_k @ homogeneous_pixel
    direction = direction.flatten()
    direction /= np.linalg.norm(direction)

    # Preserve the convention used by Geodetic before ground intersection.
    if direction[2] < 0.0:
        direction *= -1.0

    if np.isclose(direction[2], 0.0):
        raise ValueError("The pixel ray is parallel to the ground and never intersects it.")

    origin = np.asarray(drone_world_position, dtype=float).reshape(3)
    ray_parameter = -origin[2] / direction[2]
    intersection = origin + ray_parameter * direction
    intersection[2] = 0.0
    return intersection


def calculate_rows(
    intrinsic_matrix: np.ndarray, resolution: tuple[int, int]
) -> list[dict[str, object]]:
    """Calculate all combinations that will be written to the CSV file."""
    base_yaw = 0.0
    base_roll = 0.0
    image_width, image_height = resolution
    center_pixel = (image_width / 2.0, image_height / 2.0)
    rows: list[dict[str, object]] = []

    for height in HEIGHTS_M:
        drone_position = np.array([[0.0], [0.0], [height]])

        for base_pitch in PITCHES_DEGREES:
            reference_rotation = Geometry.yaw_pitch_roll_to_rotation_matrix(
                base_yaw, base_pitch, base_roll
            )
            reference_position = get_intersection_from_click(
                center_pixel, intrinsic_matrix, reference_rotation, drone_position
            )

            for angular_error in ANGULAR_ERROR_VALUES_DEGREES:
                for perturbation, affected_axes in PERTURBATIONS.items():
                    yaw_error, pitch_error, roll_error = (
                        angular_error if is_affected else 0.0
                        for is_affected in affected_axes
                    )
                    erroneous_yaw = base_yaw + yaw_error
                    erroneous_pitch = base_pitch + pitch_error
                    erroneous_roll = base_roll + roll_error
                    erroneous_rotation = Geometry.yaw_pitch_roll_to_rotation_matrix(
                        erroneous_yaw, erroneous_pitch, erroneous_roll
                    )
                    calculated_position = get_intersection_from_click(
                        center_pixel,
                        intrinsic_matrix,
                        erroneous_rotation,
                        drone_position,
                    )

                    difference = calculated_position - reference_position
                    rows.append(
                        {
                            "height_m": height,
                            "base_yaw_degrees": base_yaw,
                            "base_pitch_degrees": base_pitch,
                            "base_roll_degrees": base_roll,
                            "perturbation": perturbation,
                            "yaw_error_degrees": yaw_error,
                            "pitch_error_degrees": pitch_error,
                            "roll_error_degrees": roll_error,
                            "image_width_px": image_width,
                            "image_height_px": image_height,
                            "pixel_x": center_pixel[0],
                            "pixel_y": center_pixel[1],
                            "reference_east_m": reference_position[0],
                            "reference_north_m": reference_position[1],
                            "calculated_east_m": calculated_position[0],
                            "calculated_north_m": calculated_position[1],
                            "east_error_m": difference[0],
                            "north_error_m": difference[1],
                            "horizontal_error_m": np.linalg.norm(difference[:2]),
                        }
                    )

    return rows


def positive_integer(value: str) -> int:
    """Convert a command-line argument to a strictly positive integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("resolution values must be positive")
    return number


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Save results as a semicolon-delimited CSV file."""
    if not rows:
        raise ValueError("There are no results to save.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys()), delimiter=";"
        )
        writer.writeheader()
        writer.writerows(rows)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate position errors for +/-0.1-degree attitude perturbations."
    )
    parser.add_argument(
        "--k",
        type=Path,
        help="JSON file containing K (default: K_path from parameters.json)",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=positive_integer,
        required=True,
        metavar=("WIDTH", "HEIGHT"),
        help="image resolution in pixels, for example: --resolution 1920 1080",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("orientation_errors.csv"),
        help="output CSV file (default: orientation_errors.csv)",
    )
    return parser


def main() -> None:
    arguments = create_parser().parse_args()
    k_path = arguments.k if arguments.k is not None else get_default_k_path()
    intrinsic_matrix = load_intrinsic_matrix(k_path)
    rows = calculate_rows(intrinsic_matrix, tuple(arguments.resolution))
    save_csv(rows, arguments.output)
    print(f"CSV saved to {arguments.output.resolve()} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
