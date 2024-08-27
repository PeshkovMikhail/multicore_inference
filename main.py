import cv2
import numpy as np
from multiprocessing import Queue, Process
from yolo_tracker import YOLOv8
from speed_tracker import SpeedTracker, FRAMES_PROCESSING
from rknnlite.api import RKNNLite
from SFSORT import SFSORT
import time

images_queue = Queue()
dets_queue = Queue()
tracked_queue = Queue()
skeleton_queue = Queue()
results_queue = Queue()

def yolo_process(img_q, dets_q, core_mask):
    rknn_lite = RKNNLite()
    rknn_lite.load_rknn("../yolov8.rknn")
    rknn_lite.init_runtime(core_mask=core_mask)
    tracker = YOLOv8(rknn_lite)
    for image in iter(img_q.get, None):
        dets = tracker.run(image[0])
        dets_q.put((image[0], dets, image[1]))
    # img_q.close()
    dets_q.put(None)

def bbox_tracker(dets_raw_q, dets_q, width, height, fps):
    tracker_args = {
        "dynamic_tuning": True,
        "cth": 0.7,
        "high_th": 0.7,
        "high_th_m": 0.1,
        "match_th_first": 0.6,
        "match_th_first_m": 0.05,
        "match_th_second": 0.4,
        "low_th": 0.2,
        "new_track_th": 0.5,
        "new_track_th_m": 0.1,
        "marginal_timeout": (7 * fps // 10),
        "central_timeout": fps,
        "horizontal_margin": width // 10,
        "vertical_margin": height // 10,
        "frame_width": width,
        "frame_height": height
    }
    tracker = SFSORT(tracker_args)

    last_frame_id = -1

    temp_res_storage = {}

    for image, dets, frame_id in iter(dets_raw_q.get, None):
        if frame_id != last_frame_id + 1:
            temp_res_storage[frame_id] = (image, dets)
            continue
        bboxes = dets[:, :4].astype(np.int32)
        scores = dets[:, 5]
        tracks = np.asarray(tracker.update(bboxes, scores, np.zeros_like(scores)))
        if len(tracks) == 0:
            bboxes = []
            track_ids = []
        else:
            bboxes = tracks[:, 0]
            track_ids = tracks[:, 1]
        dets_q.put((frame_id, image, bboxes, track_ids))
        last_frame_id = frame_id
        while (last_frame_id + 1) in temp_res_storage.keys():
            last_frame_id += 1
            image, dets = temp_res_storage.pop(last_frame_id)
            bboxes = dets[:, :4].astype(np.int32)
            scores = dets[:, 5]
            tracks = np.asarray(tracker.update(bboxes, scores, np.zeros_like(scores)))
            if len(tracks) == 0:
                bboxes = []
                track_ids = []
            else:
                bboxes = tracks[:, 0]
                track_ids = tracks[:, 1]
            dets_q.put((last_frame_id, image, bboxes, track_ids))
    print("Tracker done")
    dets_raw_q.close()
    dets_q.put(None)
    dets_q.put(None)
    dets_q.put(None)
    dets_q.put(None)
    dets_q.put(None)
    dets_q.put(None)
    dets_q.put(None)
    dets_q.put(None)
    dets_q.put(None)

def pose_detector_process(tracked_q, skeleton_q, core_mask):
    rknn_lite = RKNNLite()
    rknn_lite.load_rknn("models/rtmpose_i8_V2.rknn")
    rknn_lite.init_runtime(core_mask=core_mask)

    for frame_id, image, bboxes, track_ids in iter(tracked_q.get, None):
        keypoints = np.zeros((len(bboxes), 17, 2))
        for i, bbox in enumerate(bboxes):
            try:
                person = cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (192, 256))
                raw = rknn_lite.inference(inputs=[np.expand_dims(cv2.cvtColor(person, cv2.COLOR_BGR2RGB), 1)])
                x_axis, y_axis = raw[0][0], raw[1][0]
                keypoints[i, :, 0] = np.argmax(x_axis, 1) // 2 + bbox[0]
                keypoints[i, :, 1] = np.argmax(y_axis, 1) // 2 + bbox[1]
            except:
                print(image.shape, bbox)
        skeleton_q.put((frame_id, image, bboxes, keypoints, track_ids))
    skeleton_q.put(None)
    print("Pose detector finished")
    tracked_q.close()
        


def speed_tracking_process(dets_q, res_q, height, width, fps):
    speed_tracker = SpeedTracker(height, width, 1 / fps)
    last_frame_id = -1
    # framebuffer = np.zeros((FRAMES_PROCESSING, height, width, 3))

    stored_frame_info = {}
    for frame_id, image, bboxes, keypoints, track_ids in iter(dets_q.get, None):
        dets = {}
        for i, track_id in enumerate(track_ids):
            dets[track_id] = (bboxes[i], keypoints[i])
        if frame_id - 1 != last_frame_id:
            stored_frame_info[frame_id] = (image, dets)
            continue
        frame_info = speed_tracker.speed(dets, frame_id)
        for track_id, bbox in zip(track_ids, bboxes):
            frame_info[track_id]['bbox'] = bbox
            frame_info[track_id]['label'] = 0
        res_q.put((image, frame_info))
        # if frame_id % FRAMES_PROCESSING == FRAMES_PROCESSING-1:
        #     speed_tracker.classify()
        # framebuffer[frame_id % FRAMES_PROCESSING] = image
        last_frame_id = frame_id

        while (last_frame_id + 1) in stored_frame_info.keys():
            last_frame_id += 1
            image, dets = stored_frame_info.pop(last_frame_id)
            frame_info = speed_tracker.speed(dets, last_frame_id)
            for track_id, (bbox, _) in dets.items():
                frame_info[track_id]['bbox'] = bbox
                frame_info[track_id]['label'] = 0
            res_q.put((image, frame_info))
            # if last_frame_id % FRAMES_PROCESSING == FRAMES_PROCESSING-1:
            #     speed_tracker.classify()
            # framebuffer[last_frame_id % FRAMES_PROCESSING] = image
    print("Speed done")
    res_q.put(None)


def write_process(res_q, fps, height, width):
    out_cap = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    for image, info in iter(res_q.get, None):
        for track_id, track_info in info.items():
            color = (255, 0, 0)
            if track_info['label'] == 1:
                color = (0, 255, 0)
            bbox = track_info['bbox']
            speed = track_info['speed']
            cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
            cv2.putText(image, f"{track_id} {speed:.2f} km/h", bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        out_cap.write(image)

    out_cap.release()
    print("Writing Done")

if __name__ == '__main__':
    cap = cv2.VideoCapture("../output000.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    threads = 12
    yolos_per_thread = threads // len(core_masks)
    yolos = [Process(target=yolo_process, args=(images_queue, dets_queue, core_masks[t//yolos_per_thread])) for t in range(threads)]
    for y in yolos:
        y.start()
    tracker = Process(target = bbox_tracker, args=(dets_queue, tracked_queue, width, height, fps//2))
    pose_detectors = [Process(target=pose_detector_process, args = (tracked_queue, skeleton_queue, core_masks[t//yolos_per_thread])) for t in range(threads)]
    for p in pose_detectors:
        p.start()
    tracker.start()
    speed = Process(target=speed_tracking_process, args=(skeleton_queue, results_queue, height, width, fps//2))
    speed.start()
    writer = Process(target=write_process, args=(results_queue, 15, height, width))
    writer.start()

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % 2 != 0:
            frame_id += 1
            continue
        # while images_queue.qsize() > 100:
        #     print(images_queue.qsize())
        #     time.sleep(0.5)
        images_queue.put((frame, frame_id//2))
        frame_id += 1

    for i in range(threads):
        images_queue.put(None)
    
    for y in yolos:
        y.join()
    tracker.join()
    for p in pose_detectors:
        p.join()
    
    speed.join()
    writer.join()
    print("Done")