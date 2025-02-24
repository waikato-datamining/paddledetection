import os
import yaml
from datetime import datetime
from typing import List

import paddle
from deploy.python.infer import Detector, DetectorSOLOv2, DetectorPicoDet, DetectorCLRNet
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from smu import mask_to_polygon, polygon_to_lists
from shapely.geometry import Polygon as SPolygon


def load_model(model_path: str, device: str = "gpu", threshold: float = 0.5) -> Detector:
    """
    Loads the model and instantiates the appropriate detector.

    :param model_path: the path with the inference model
    :type model_path: str
    :param device: the device to run the model on, e.g., gpu or cpu
    :type device: str
    :param threshold: the minimum score for predictions
    :type threshold: float
    :return: the detector instance
    :rtype: Detector
    """
    paddle.enable_static()
    fd_deploy_file = os.path.join(model_path, 'inference.yml')
    ppdet_deploy_file = os.path.join(model_path, 'infer_cfg.yml')
    use_fd_format = os.path.exists(fd_deploy_file)
    if use_fd_format:
        deploy_file = fd_deploy_file
    else:
        deploy_file = ppdet_deploy_file
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    if arch == 'SOLOv2':
        detector = DetectorSOLOv2(model_path, device=device.upper(), threshold=threshold, use_fd_format=use_fd_format)
    elif arch == 'PicoDet':
        detector = DetectorPicoDet(model_path, device=device.upper(), threshold=threshold, use_fd_format=use_fd_format)
    elif arch == "CLRNet":
        detector = DetectorCLRNet(model_path, device=device.upper(), threshold=threshold, use_fd_format=use_fd_format)
    else:
        detector = Detector(model_path, device=device.upper(), threshold=threshold, use_fd_format=use_fd_format)
    return detector


def load_label_list(path: str) -> List[str]:
    """
    Loads the comma-separated list of labels from the specified path.

    :param path: the file with the label list
    :type path: str
    :return: the list of labels
    :rtype: list
    """
    with open(path, "r") as fp:
        result = fp.readline().strip().split(",")
    return result


def prediction_to_file(predictions, labels, id_: str, path: str, threshold: float = 0.5, mask_nth: int = 1) -> str:
    """
    Saves the predictions as OPEX in the specified file. 

    :param predictions: the predictions to save
    :param labels: the list of labels
    :param id_: the ID for the OPEX output
    :type id_: str
    :param path: the file to save the predictions to
    :type path: str
    :param threshold: the minimum score for retaining predictions
    :type threshold: float
    :param mask_nth: to speed polygon detection up, use every nth row and column only (instance segmentation only), use < 1 to turn off polygon calculation
    :type mask_nth: int
    :return: the file the predictions were saved to
    :rtype: str
    """
    data = prediction_to_data(predictions, labels, id_, threshold=threshold, mask_nth=mask_nth)
    with open(path, "w") as fp:
        fp.write(data)
        fp.write("\n")
    return path


def prediction_to_data(predictions, labels, id_: str, threshold: float = 0.5, mask_nth: int = 1) -> str:
    """
    Turns the predictions into an OPEX string.

    :param predictions: the predictions to convert
    :param labels: the list of labels
    :param id_: the ID for the OPEX output
    :type id_: str
    :param threshold: the minimum score for retaining predictions
    :type threshold: float
    :param mask_nth: to speed polygon detection up, use every nth row and column only (instance segmentation only), use < 1 to turn off polygon calculation
    :type mask_nth: int
    :return: the generated predictions
    :rtype: str
    """
    pred_objs = []
    for i, box in enumerate(predictions['boxes']):
        score = float(box[1])
        if score < threshold:
            continue
        labelid = int(box[0])
        xmin = int(box[2])
        ymin = int(box[3])
        xmax = int(box[4])
        ymax = int(box[5])
        bbox = BBox(left=xmin, top=ymin, right=xmax, bottom=ymax)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        if ('masks' in predictions) and (mask_nth > 0):
            mask = predictions['masks'][i]
            polys = mask_to_polygon(mask, mask_nth=mask_nth)  # determine polygons
            if len(polys) > 0:
                # find largest polygon
                area = 0.0
                for poly in polys:
                    px, py = polygon_to_lists(poly, swap_x_y=True, as_type="int")  # get coordinates
                    area_curr = SPolygon(zip(px, py)).area
                    if area_curr > area:
                        area = area_curr
                        points = [(x, y) for x, y in zip(px, py)]
        poly = Polygon(points=points)
        label = labels[labelid]
        opex_obj = ObjectPrediction(label=label, score=score, bbox=bbox, polygon=poly)
        pred_objs.append(opex_obj)
    preds = ObjectPredictions(id=id_, timestamp=str(datetime.now()), objects=pred_objs)
    return preds.to_json_string()

