# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv

sys.path.append("E:\hr\work\segment-anything-main")
print(sys.path)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import argparse
import json

from typing import Any, Dict, List

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    help="Path to either a single input image or folder of images.",
    default=r"E:\hr\dataset\test.png",
)

parser.add_argument(
    "--input_path",
    type=str,
    help="Path to either a single input image or folder of images.",
    default=r"E:\hr\data_test\SAM\input"
)

parser.add_argument(
    "--output",
    type=str,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
    default=r'E:\hr\dataset',
)

parser.add_argument(
    "--output_path_all",
    type=str,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
    default=r"E:\hr\data_test\SAM\output\seg_all"
)

parser.add_argument(
    "--output_path_pro",
    type=str,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
    default=r"E:\hr\data_test\SAM\output\seg_pro"
)

parser.add_argument(
    "--model-type",
    type=str,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    default='default',
)

parser.add_argument(
    "--checkpoint",
    type=str,
    help="The path to the SAM checkpoint to use for mask generation.",
    default='E:\hr\CK\sam_vit_h_4b8939.pth',
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def write_masks_to_folder(image, masks: List[Dict[str, Any]], path: str, use_prompt) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    # save img
    img = Image.fromarray(image)
    os.makedirs(fr'{path}', exist_ok=True)
    img.save(fr'{path}\img.png')

    for i, mask_data in enumerate(masks):
        # 输入提示时改变掩码数据只能通过索引读取
        if use_prompt:
            mask = mask_data
        else:
            mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)

        # 将掩码数组的形状转换为与原始图像相同
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # np.expand_dims(mask, axis=2)



        # save img_crop by mask
        cropped_image = Image.fromarray(np.uint8(image * mask))
        os.makedirs(fr'{path}\crop', exist_ok=True)
        cropped_image.save(fr'{path}\crop\{i}.png')

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "output_path_all": args.output_path_all,
        "output_path_pro": args.output_path_pro,
        "input_path": args.input_path,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry["default"](checkpoint="E:\hr\CK\sam_vit_h_4b8939.pth")
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    args.input = args.input_path
    use_prompt = False
    if use_prompt:
        args.output = args.output_path_pro
        predictor = SamPredictor(sam)
    else:
        args.output = args.output_path_all
        generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        p = t
        p = p.split('\\')[-1].split('.')[0]
        if use_prompt:
            input_point = np.array([[320, 240], [350, 260]])
            input_label = np.array([0, 1])
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label,
                                                 multimask_output=True)
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(mask, plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
                plt.axis("on")
                plt.show()

        else:
            masks = generator.generate(image)

            mask_annotator = sv.MaskAnnotator()
            detections = sv.Detections.from_sam(masks)
            annotated_image = mask_annotator.annotate(image, detections)

            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_image)
            plt.show()

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            write_masks_to_folder(image, masks, save_base, use_prompt=use_prompt)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
