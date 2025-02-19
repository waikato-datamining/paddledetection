import os
import yaml
from datetime import datetime

import paddle
from deploy.python.infer import Detector, DetectorSOLOv2, DetectorPicoDet, DetectorCLRNet
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon


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


def prediction_to_file(predictions, labels, id_: str, path: str, threshold: float = 0.5) -> str:
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
    :return: the file the predictions were saved to
    :rtype: str
    """
    data = prediction_to_data(predictions, labels, id_, threshold=threshold)
    with open(path, "w") as fp:
        fp.write(data)
        fp.write("\n")
    return path


def prediction_to_data(predictions, labels, id_: str, threshold: float = 0.5) -> str:
    """
    Turns the predictions into an OPEX string.

    :param predictions: the predictions to convert
    :param labels: the list of labels
    :param id_: the ID for the OPEX output
    :type id_: str
    :param threshold: the minimum score for retaining predictions
    :type threshold: float
    :return: the generated predictions
    :rtype: str
    """
    pred_objs = []
    for box in predictions['boxes']:
        score = float(box[1])
        if score < threshold:
            continue
        labelid = int(box[0])
        xmin = int(box[2])
        ymin = int(box[3])
        xmax = int(box[4])
        ymax = int(box[5])
        bbox = BBox(left=xmin, top=ymin, right=xmax, bottom=ymax)
        poly = Polygon(points=[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        label = labels[labelid]
        opex_obj = ObjectPrediction(label=label, score=score, bbox=bbox, polygon=poly)
        pred_objs.append(opex_obj)
    preds = ObjectPredictions(id=id_, timestamp=str(datetime.now()), objects=pred_objs)
    return preds.to_json_string()

