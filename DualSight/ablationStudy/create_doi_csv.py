"""
create_doi_csv.py

Generates a 'DOI.csv' (Design of Experiments) with all combinations of:
- ID
- Model
- BoxInclusion
- MaskInclusion
- NumberOfPOIs
- POIPlacementAlgorithm
- PerimeterBuffer
- BoundingBoxDistortion (conditional on BoxInclusion)
- MaskDistortion (conditional on MaskInclusion)
"""

import csv
import itertools

def create_doi_csv(csv_filename="DOI.csv"):
    # Define categories/conditions:
    models = [
        "YOLOv8 Nano",
        "YOLOv8 XL",
        "Mask R-CNN",
        "Mask2Former",
        "YOLOv8 Nano + Sam"
        # "MobileNetV3"
    ]
    box_inclusion = [True, False]
    mask_inclusion = [True, False]
    number_of_pois = [1, 2, 3, 4, 5, 6, 7]
    poi_placement_alg = ["Random", "Distance Max", "Voronoi"]
    perimeter_buffer = ["0%", "5%", "10%", "15%"]
    bounding_box_distortion = ["90%", "100%", "110%"]
    mask_distortion = ["90%", "100%", "110%"]

    # Columns for CSV (adding "ID" as the first column)
    columns = [
        "ID",
        "Model",
        "BoxInclusion",
        "MaskInclusion",
        "NumberOfPOIs",
        "POIPlacementAlgorithm",
        "PerimeterBuffer",
        "BoundingBoxDistortion",
        "MaskDistortion"
    ]

    rows = []
    id_counter = 1  # Initialize ID counter

    # Iterate over base combinations without distortions
    base_combinations = itertools.product(
        models,
        box_inclusion,
        mask_inclusion,
        number_of_pois,
        poi_placement_alg,
        perimeter_buffer
    )

    for (model, box_inc, mask_inc, num_pois, poi_alg, buffer_pct) in base_combinations:
        # Determine applicable distortions based on inclusion flags
        bb_dists = bounding_box_distortion if box_inc else ["N/A"]
        mask_dists = mask_distortion if mask_inc else ["N/A"]

        # Create rows for each combination of distortions
        for bb_dist in bb_dists:
            for m_dist in mask_dists:
                row = (
                    id_counter,
                    model,
                    box_inc,
                    mask_inc,
                    num_pois,
                    poi_alg,
                    buffer_pct,
                    bb_dist,
                    m_dist
                )
                rows.append(row)
                id_counter += 1  # Increment ID for next row

    # Write rows to CSV
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)  # Write header
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Created {csv_filename} with {len(rows)} experimental conditions.")

if __name__ == "__main__":
    create_doi_csv("DOI.csv")
