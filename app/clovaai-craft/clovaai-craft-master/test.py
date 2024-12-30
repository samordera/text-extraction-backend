from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import torch
from utils import craft_utils, file_utils, imgproc
from nets.nn import CRAFT, RefineNet
import yaml
import os
from collections import OrderedDict
from typing import List
import pytesseract
import traceback

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

with open(os.path.join('utils', 'config.yaml')) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

image_list, _, _ = file_utils.get_files(args['test_folder'])

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def read_coordinates(filename, x_padding=5, y_padding=2):
    with open(filename, 'r') as file:
        lines = file.readlines()
        coords = []
        for line in lines:
            line = line.strip()
            if line:
                coord_list = list(map(float, line.split(',')))
                if len(coord_list) % 2 == 0:
                    x_coords = coord_list[0::2]
                    y_coords = coord_list[1::2]
                    min_x, max_x = min(x_coords) - x_padding, max(x_coords) + x_padding
                    min_y, max_y = min(y_coords) - y_padding, max(y_coords) + y_padding
                    padded_coords = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
                    coords.append(padded_coords)
    return coords

def convert_to_bbox_format(box):
    """
    Convert a 4-point box format to [x_min, y_min, x_max, y_max].
    """
    box = np.array(box)
    x_min = np.min(box[:, 0])
    y_min = np.min(box[:, 1])
    x_max = np.max(box[:, 0])
    y_max = np.max(box[:, 1])
    return [x_min, y_min, x_max, y_max]

def overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return not (
        x1_max < x2_min or x2_max < x1_min or
        y1_max < y2_min or y2_max < y1_min
    )


def merge_boxes(box1, box2):
    """
    Merge two bounding boxes into a single bounding box.
    """
    box1 = [float(x) for x in box1]
    box2 = [float(x) for x in box2]

    if len(box1) != 4 or len(box2) != 4:
        raise ValueError(f"Invalid bounding box format: box1={box1}, box2={box2}")

    # Merge the boxes
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    
    return [x_min, y_min, x_max, y_max]

def merge_overlapping_boxes(coords):
    """
    Merges overlapping bounding boxes.
    """
    coords = [convert_to_bbox_format(box) for box in coords]
    merged_boxes = []
    while coords:
        box1 = coords.pop(0)
        to_merge = []
        for box2 in coords:
            if overlap(box1, box2):
                to_merge.append(box2)
        # Merge all overlapping boxes into one
        for box in to_merge:
            coords.remove(box)
            box1 = merge_boxes(box1, box)
        merged_boxes.append(box1)
    return merged_boxes



def extract_text_from_image(image, coords):
    extracted_texts = []
    for box in coords:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cropped)
        extracted_texts.append(text)
    return extracted_texts


def draw_bounding_boxes(image, coords):
    for box in coords:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(os.path.join('utils', 'config.yaml')) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

net = CRAFT()
refine_net = None

if torch.cuda.is_available() and args['cuda']:
    net.load_state_dict(copyStateDict(torch.load(args['trained_model'])), strict=False)
    net = net.cuda()
    net = torch.nn.DataParallel(net)
else:
    net.load_state_dict(copyStateDict(torch.load(args['trained_model'], map_location=torch.device('cpu'))), strict=False)

net.eval()

if args.get('refine', False):
    refine_net = RefineNet()
    if torch.cuda.is_available() and args['cuda']:
        refine_net.load_state_dict(copyStateDict(torch.load(args['refiner_model'])), strict=False)
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args['refiner_model'], map_location=torch.device('cpu'))), strict=False)
        refine_net = refine_net.cpu()
    refine_net.eval()

def process_image(image: np.ndarray):
    bboxes, polys, _ = test_net(
        net, image, args['text_threshold'], args['link_threshold'],
        args['low_text'], torch.cuda.is_available() and args['cuda'],
        args['poly'], refine_net
    )
    
    bboxes = [bbox.tolist() for bbox in bboxes] if isinstance(bboxes, np.ndarray) else bboxes

    merged_coords = merge_overlapping_boxes(bboxes)

    texts = extract_text_from_image(image, merged_coords)

    return texts

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, args['canvas_size'], interpolation=cv2.INTER_LINEAR, mag_ratio=args['mag_ratio']
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    polys = [box.reshape(-1).tolist() for box in boxes]

    return boxes, polys, None

@app.get("/")
async def home():
    return JSONResponse(content={
        "success": True,
        "message": "Server is up..."
    })

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        texts = process_image(image_bgr)
        print(texts)

        bboxes, polys, _ = test_net(
            net, image_bgr, args['text_threshold'], args['link_threshold'],
            args['low_text'], torch.cuda.is_available() and args['cuda'],
            args['poly'], refine_net
        )
        merged_coords = merge_overlapping_boxes(bboxes)
        boxed_image = draw_bounding_boxes(image_bgr.copy(), merged_coords)

        _, img_encoded = cv2.imencode('.png', boxed_image)
        image_bytes = img_encoded.tobytes()

        return JSONResponse(content={"extracted_texts": texts})
    except Exception as e:
            error_trace = traceback.format_exc()
            print(error_trace)
            return JSONResponse(content={"error": str(e), "traceback": error_trace}, status_code=500)

