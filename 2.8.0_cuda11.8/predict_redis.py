from datetime import datetime
import numpy as np
import traceback
import cv2

from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log
from predict_common import load_model, load_label_list, prediction_to_data


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        array = np.frombuffer(msg_cont.message['data'], np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_COLOR)
        preds = config.detector.predict_image([img], visual=False)
        labels = config.detector.labels if (config.labels is None) else config.labels
        out_data = prediction_to_data(preds, labels, str(start_time), config.threshold)
        msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser('PaddleDetection - Prediction (Redis)', prog="paddledet_predict_redis", prefix="redis_")
    parser.add_argument('--model_path', help='Path to the exported inference model', required=True, default=None)
    parser.add_argument('--label_list', help='Path to the file with the comma-separated list of labels to override model-internal ones', required=False, default=None)
    parser.add_argument('--device', help='The device to use', default="gpu")
    parser.add_argument('--threshold', help='The score threshold for predictions', required=False, default=0.5)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        detector = load_model(parsed.model_path, device=parsed.device.upper(), threshold=parsed.threshold)
        labels = None if (parsed.label_list is None) else load_label_list(parsed.label_list)

        config = Container()
        config.detector = detector
        config.labels = labels
        config.threshold = parsed.threshold
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())

