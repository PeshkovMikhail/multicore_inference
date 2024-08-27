from concurrent.futures import ThreadPoolExecutor
from rknnlite.api import RKNNLite
from yolo_tracker import YOLOv10
import queue
import cv2

THREADS = 3

def init_model(core_mask):
    rknn_lite = RKNNLite()
    rknn_lite.load_rknn("../yolo10.rknn")
    rknn_lite.init_runtime(core_mask=core_mask)
    
    return YOLOv10(rknn_lite)

def inference(frame, model, output_queue):
    output_queue.put(model.run(frame))

if __name__ == '__main__':
    core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    yolos_per_thread = THREADS // len(core_masks)
    yolos = [init_model(core_masks[t//yolos_per_thread]) for t in range(THREADS)]

    frame_id = 0

    cap = cv2.VideoCapture("../output000.mp4")
    output_queue = queue.Queue()
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            executor.submit(inference, frame, yolos[frame_id % THREADS], output_queue)
            frame_id += 1
        executor.shutdown()


