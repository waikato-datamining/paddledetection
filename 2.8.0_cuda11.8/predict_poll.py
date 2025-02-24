import os
import argparse
import traceback

from image_complete import auto
from sfp import Poller
from predict_common import load_model, load_label_list, prediction_to_file

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []

    try:
        import cv2
        img = cv2.imread(fname, 1)
        preds = poller.params.detector.predict_image([img], visual=False)
        fname_out = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ".json")
        labels = poller.params.detector.labels if (poller.params.labels is None) else poller.params.labels
        fname_out = prediction_to_file(preds, labels, os.path.basename(fname), fname_out,
                                       threshold=poller.params.threshold, mask_nth=poller.params.mask_nth)
        result.append(fname_out)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(detector, input_dir, output_dir, tmp_dir, labels=None, threshold=0.5, mask_nth: int = 1,
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, verbose=False, quiet=False):
    """
    Method for performing predictions on images.

    :param detector: the detector to use
    :type detector: Detector
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished, use None if not to use
    :type tmp_dir: str
    :param labels: the list of labels to use for overriding the model internal ones, ignore if None
    :type labels: list or None
    :param threshold: the score threshold to use
    :type threshold: float
    :param mask_nth: to speed polygon detection up, use every nth row and column only (instance segmentation only)
    :type mask_nth: int
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.progress = not quiet
    poller.verbose = verbose
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.detector = detector
    poller.params.labels = labels
    poller.params.threshold = threshold
    poller.params.mask_nth = mask_nth
    poller.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PaddleDetection - Prediction", prog="paddledet_predict_poll", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', help='Path to the exported inference model', required=True, default=None)
    parser.add_argument('--label_list', help='Path to the file with the comma-separated list of labels to override model-internal ones', required=False, default=None)
    parser.add_argument('--device', help='The device to use', default="gpu")
    parser.add_argument('--threshold', type=float, help='The score threshold for predictions', required=False, default=0.5)
    parser.add_argument('--mask_nth', type=int, help='To speed polygon detection up, use every nth row and column only; use < 1 to turn off polygon calculation (instance segmentation only)', required=False, default=1)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args()

    try:
        detector = load_model(parsed.model_path, device=parsed.device.upper(), threshold=parsed.threshold)
        labels = None if (parsed.label_list is None) else load_label_list(parsed.label_list)

        # Performing the prediction and producing the predictions files
        predict_on_images(detector, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp,
                          labels=labels, threshold=parsed.threshold, mask_nth=parsed.mask_nth, continuous=parsed.continuous,
                          use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                          delete_input=parsed.delete_input, verbose=parsed.verbose, quiet=parsed.quiet)

    except Exception as e:
        print(traceback.format_exc())

