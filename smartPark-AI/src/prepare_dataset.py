import json
import os
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("raw_data")
OUTPUT_DIR = Path("dataset")

SPLITS = ["train", "valid", "test"]

# Padding around each parking-space bounding box
PADDING = 5

# Minimum crop size
MIN_SIZE = 20


# ============================================================
# CLEAN OLD DATASET
# ============================================================

if OUTPUT_DIR.exists():
    print("Removing old dataset...")
    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "valid", "test"]:

    for class_name in ["occupied", "vacant"]:

        directory = OUTPUT_DIR / split / class_name

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

total_crops = 0

for split in SPLITS:

    print("\n" + "=" * 60)
    print(f"Processing {split.upper()}")
    print("=" * 60)

    annotation_file = (
        RAW_DIR
        / split
        / "_annotations.coco.json"
    )

    if not annotation_file.exists():

        print(
            f"Annotation file not found: {annotation_file}"
        )

        continue


    # --------------------------------------------------------
    # LOAD COCO JSON
    # --------------------------------------------------------

    with open(
        annotation_file,
        "r"
    ) as f:

        coco = json.load(f)


    # --------------------------------------------------------
    # CATEGORY INFORMATION
    # --------------------------------------------------------

    categories = {
        category["id"]: category["name"].lower().strip()
        for category in coco["categories"]
    }

    print("\nCategories:")

    for category_id, category_name in categories.items():

        print(
            f"  {category_id}: {category_name}"
        )


    # --------------------------------------------------------
    # FIND OCCUPIED / VACANT CATEGORY IDS
    # --------------------------------------------------------

    occupied_ids = set()
    vacant_ids = set()

    for category_id, category_name in categories.items():

        if "occup" in category_name:

            occupied_ids.add(category_id)

        elif (
            "vacant" in category_name
            or "empty" in category_name
            or "free" in category_name
        ):

            vacant_ids.add(category_id)


    print("\nOccupied IDs:", occupied_ids)
    print("Vacant IDs:", vacant_ids)


    if not occupied_ids or not vacant_ids:

        print(
            "\nERROR: Could not automatically identify "
            "occupied/vacant categories."
        )

        print(
            "Please send me the Categories output above."
        )

        raise SystemExit


    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    images = {
        image["id"]: image
        for image in coco["images"]
    }


    # --------------------------------------------------------
    # ANNOTATIONS GROUPED BY IMAGE
    # --------------------------------------------------------

    annotations_by_image = {}

    for annotation in coco["annotations"]:

        image_id = annotation["image_id"]

        if image_id not in annotations_by_image:

            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(
            annotation
        )


    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    split_crops = 0

    for index, (image_id, image_info) in enumerate(
        images.items(),
        start=1
    ):

        image_filename = image_info["file_name"]

        image_path = (
            RAW_DIR
            / split
            / image_filename
        )


        if not image_path.exists():

            print(
                f"WARNING: Image not found: {image_path}"
            )

            continue


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"Could not open {image_path}: {e}"
            )

            continue


        image_width, image_height = image.size


        # ----------------------------------------------------
        # GET ANNOTATIONS
        # ----------------------------------------------------

        annotations = annotations_by_image.get(
            image_id,
            []
        )


        for annotation_index, annotation in enumerate(
            annotations
        ):

            category_id = annotation["category_id"]


            # ------------------------------------------------
            # DETERMINE CLASS
            # ------------------------------------------------

            if category_id in occupied_ids:

                class_name = "occupied"

            elif category_id in vacant_ids:

                class_name = "vacant"

            else:

                continue


            # ------------------------------------------------
            # COCO BOUNDING BOX
            #
            # [x, y, width, height]
            # ------------------------------------------------

            bbox = annotation.get(
                "bbox"
            )

            if not bbox or len(bbox) != 4:

                continue


            x, y, width, height = bbox


            # ------------------------------------------------
            # CONVERT TO INTEGER COORDINATES
            # ------------------------------------------------

            x1 = int(x) - PADDING
            y1 = int(y) - PADDING

            x2 = int(
                x + width
            ) + PADDING

            y2 = int(
                y + height
            ) + PADDING


            # ------------------------------------------------
            # CLAMP TO IMAGE BOUNDARIES
            # ------------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                image_width,
                x2
            )

            y2 = min(
                image_height,
                y2
            )


            # ------------------------------------------------
            # VALIDATE CROP
            # ------------------------------------------------

            crop_width = x2 - x1
            crop_height = y2 - y1

            if (
                crop_width < MIN_SIZE
                or crop_height < MIN_SIZE
            ):

                continue


            # ------------------------------------------------
            # CROP PARKING SPACE
            # ------------------------------------------------

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            # ------------------------------------------------
            # CREATE OUTPUT NAME
            # ------------------------------------------------

            original_name = Path(
                image_filename
            ).stem

            output_name = (
                f"{original_name}"
                f"_slot_{annotation_index}"
                f".jpg"
            )


            output_path = (
                OUTPUT_DIR
                / split
                / class_name
                / output_name
            )


            # ------------------------------------------------
            # SAVE CROP
            # ------------------------------------------------

            crop.save(
                output_path,
                quality=95
            )

            split_crops += 1
            total_crops += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if index % 500 == 0:

            print(
                f"Processed {index}/{len(images)} images..."
            )


    # --------------------------------------------------------
    # SPLIT SUMMARY
    # --------------------------------------------------------

    occupied_count = len(
        list(
            (
                OUTPUT_DIR
                / split
                / "occupied"
            ).glob("*.jpg")
        )
    )

    vacant_count = len(
        list(
            (
                OUTPUT_DIR
                / split
                / "vacant"
            ).glob("*.jpg")
        )
    )


    print(
        f"\n{split.upper()} completed."
    )

    print(
        f"Parking-space crops: {split_crops}"
    )

    print(
        f"Occupied: {occupied_count}"
    )

    print(
        f"Vacant: {vacant_count}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    f"Total parking-space crops: {total_crops}"
)


for split in ["train", "valid", "test"]:

    occupied_dir = (
        OUTPUT_DIR
        / split
        / "occupied"
    )

    vacant_dir = (
        OUTPUT_DIR
        / split
        / "vacant"
    )

    occupied_count = len(
        list(
            occupied_dir.glob("*.jpg")
        )
    )

    vacant_count = len(
        list(
            vacant_dir.glob("*.jpg")
        )
    )

    print(
        f"\n{split}:"
    )

    print(
        f"  Occupied: {occupied_count}"
    )

    print(
        f"  Vacant:   {vacant_count}"
    )