import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

LEFT_IMAGE_PATH = "l.jpeg"
RIGHT_IMAGE_PATH = "r.jpeg"

# Assumptions
#TILE_SIZE_FEET = 1.0 #ft
#BASELINE_TILES = 2.0 #ft
BASELINE_METERS = 0.15#TILE_SIZE_FEET * 0.3048 * BASELINE_TILES   # 0.6096 m

# Approximate focal length in pixels
FOCAL_LENGTH_PIXELS = 1500.0

# Principal point = image center
# This will be computed automatically after loading image.

OUTPUT_CSV = "classroom_xy_coordinates.csv"
OUTPUT_PLOT = "classroom_2d_plot.png"

# ============================================================
# GLOBALS FOR CLICKING
# ============================================================

left_points = []
right_points = []
object_info = []   # list of tuples: (object_name, object_type)

current_mode = "left"
display_image = None
display_clone = None

table_count = 1
chair_count = 1

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def draw_points(img, points, labels):
    out = img.copy()
    for i, ((x, y), (name, obj_type)) in enumerate(zip(points, labels)):
        color = (0, 0, 255) if obj_type == "table" else (255, 0, 0)  # BGR
        cv2.circle(out, (x, y), 6, color, -1)
        cv2.putText(out, name, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def mouse_callback_left(event, x, y, flags, param):
    global left_points, object_info, table_count, chair_count, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        obj = input("Enter object type for this point (t = table, c = chair): ").strip().lower()

        if obj == "t":
            name = f"T{table_count}"
            obj_type = "table"
            table_count += 1
        elif obj == "c":
            name = f"C{chair_count}"
            obj_type = "chair"
            chair_count += 1
        else:
            print("Invalid type. Use only 't' or 'c'. Point ignored.")
            return

        left_points.append((x, y))
        object_info.append((name, obj_type))
        print(f"Added LEFT point: {name} ({obj_type}) at ({x}, {y})")

        display_image = draw_points(display_clone, left_points, object_info)
        cv2.imshow("Left Image - Click objects", display_image)


def mouse_callback_right(event, x, y, flags, param):
    global right_points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(right_points) >= len(object_info):
            print("All right-image points already selected.")
            return

        idx = len(right_points)
        name, obj_type = object_info[idx]
        right_points.append((x, y))
        print(f"Added RIGHT point: {name} ({obj_type}) at ({x}, {y})")

        display_image = draw_points(display_clone, right_points, object_info[:len(right_points)])
        cv2.imshow("Right Image - Click corresponding objects in SAME order", display_image)


def compute_coordinates(left_pts, right_pts, labels, f, cx, baseline):
    results = []

    for i, (((uL, vL), (uR, vR)), (name, obj_type)) in enumerate(zip(zip(left_pts, right_pts), labels)):
        disparity = float(uL - uR)

        if disparity <= 1.0:
            print(f"Skipping {name}: disparity too small or invalid ({disparity:.2f})")
            continue

        Z = (f * baseline) / disparity
        X = ((uL - cx) * Z) / f

        results.append({
            "name": name,
            "type": obj_type,
            "uL": uL,
            "vL": vL,
            "uR": uR,
            "vR": vR,
            "disparity": disparity,
            "X_m": X,
            "Y_m": Z   # using depth as floor-plane Y
        })

    return results


def save_csv(results, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "type", "uL", "vL", "uR", "vR", "disparity", "X_m", "Y_m"])
        for r in results:
            writer.writerow([
                r["name"], r["type"],
                r["uL"], r["vL"], r["uR"], r["vR"],
                f"{r['disparity']:.3f}",
                f"{r['X_m']:.3f}",
                f"{r['Y_m']:.3f}"
            ])


def make_plot(results, plot_path):
    plt.figure(figsize=(8, 6))

    table_plotted = False
    chair_plotted = False

    for r in results:
        x = r["X_m"]
        y = r["Y_m"]
        name = r["name"]

        if r["type"] == "table":
            plt.scatter(x, y, c="red", s=100, label="Table" if not table_plotted else "")
            plt.text(x + 0.03, y + 0.03, name, color="red", fontsize=9)
            table_plotted = True
        else:
            plt.scatter(x, y, c="blue", s=100, label="Chair" if not chair_plotted else "")
            plt.text(x + 0.03, y + 0.03, name, color="blue", fontsize=9)
            chair_plotted = True

    plt.xlabel("X position (meters)")
    plt.ylabel("Y position / depth (meters)")
    plt.title("2D Floor Map of Classroom Tables and Chairs")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    global display_image, display_clone

    left_img = cv2.imread(LEFT_IMAGE_PATH)
    right_img = cv2.imread(RIGHT_IMAGE_PATH)

    if left_img is None:
        print(f"Could not read left image: {LEFT_IMAGE_PATH}")
        return
    if right_img is None:
        print(f"Could not read right image: {RIGHT_IMAGE_PATH}")
        return

    h, w = left_img.shape[:2]
    cx = w / 2.0

    print("\n================ LEFT IMAGE SELECTION ================")
    print("Instructions:")
    print("1. Click one object center at a time on the LEFT image.")
    print("2. After each click, type:")
    print("      t  -> table")
    print("      c  -> chair")
    print("3. Press 'q' when finished selecting all LEFT points.\n")

    display_clone = left_img.copy()
    display_image = left_img.copy()

    cv2.namedWindow("Left Image - Click objects", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Left Image - Click objects", mouse_callback_left)

    while True:
        cv2.imshow("Left Image - Click objects", display_image)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow("Left Image - Click objects")

    if len(left_points) == 0:
        print("No points selected in left image.")
        return

    print("\nSelected LEFT objects:")
    for i, ((x, y), (name, obj_type)) in enumerate(zip(left_points, object_info), start=1):
        print(f"{i}. {name} ({obj_type}) -> ({x}, {y})")

    print("\n================ RIGHT IMAGE SELECTION ================")
    print("Instructions:")
    print("1. Click the SAME objects in the RIGHT image.")
    print("2. Click in EXACTLY the same order as the LEFT image.")
    print("3. Press 'q' when finished.\n")

    display_clone = right_img.copy()
    display_image = right_img.copy()

    cv2.namedWindow("Right Image - Click corresponding objects in SAME order", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Right Image - Click corresponding objects in SAME order", mouse_callback_right)

    while True:
        cv2.imshow("Right Image - Click corresponding objects in SAME order", display_image)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(right_points) != len(left_points):
        print(f"Mismatch: {len(left_points)} left points but {len(right_points)} right points.")
        print("Please run again and select equal number of points.")
        return

    results = compute_coordinates(
        left_pts=left_points,
        right_pts=right_points,
        labels=object_info,
        f=FOCAL_LENGTH_PIXELS,
        cx=cx,
        baseline=BASELINE_METERS
    )

    if len(results) == 0:
        print("No valid stereo matches found.")
        return

    print("\n================ COMPUTED COORDINATES ================")
    for r in results:
        print(f"{r['name']} ({r['type']}): "
              f"disparity={r['disparity']:.2f}, "
              f"X={r['X_m']:.3f} m, Y={r['Y_m']:.3f} m")

    save_csv(results, OUTPUT_CSV)
    make_plot(results, OUTPUT_PLOT)

    print("\nDone.")
    print(f"Saved CSV  : {OUTPUT_CSV}")
    print(f"Saved Plot : {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()