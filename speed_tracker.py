import numpy as np
import cv2

NOSE:           int = 0
LEFT_EYE:       int = 1
RIGHT_EYE:      int = 2
LEFT_EAR:       int = 3
RIGHT_EAR:      int = 4
LEFT_SHOULDER:  int = 5
RIGHT_SHOULDER: int = 6
LEFT_ELBOW:     int = 7
RIGHT_ELBOW:    int = 8
LEFT_WRIST:     int = 9
RIGHT_WRIST:    int = 10
LEFT_HIP:       int = 11
RIGHT_HIP:      int = 12
LEFT_KNEE:      int = 13
RIGHT_KNEE:     int = 14
LEFT_ANKLE:     int = 15
RIGHT_ANKLE:    int = 16

HEAD_POINTS = [NOSE, RIGHT_EAR, RIGHT_EYE, LEFT_EAR, LEFT_EYE]
CHEST_POINTS = [RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_HIP, LEFT_HIP]

TRACK_POINTS = HEAD_POINTS
FRAMES_PROCESSING = 40

SPEED_WINDOW = 10
FPS_DIVIDER = 1

HEIGHT_SECTORS = 10
WIDTH_SECTORS = 20

AVERAGE_HEIGHT = 1.6


class TrackData:
    def __init__(self, frame_id, average_height_sectors, sector_len, period):
        self.height = 0
        self.height_count = 0
        self.speed_data = np.zeros((SPEED_WINDOW, 2))
        self.average_height_sectors = average_height_sectors
        self.period = period
        self.cursor = 0
        self.frame_id = frame_id
        self.prev_coords = None
        self.sector_len = sector_len

    def update_speed(self, bbox, kp, frame_id):
        coords = get_coords(kp)
        pixel_height = get_pixel_height(kp)

        if np.isnan(pixel_height):
            if np.any(np.isnan(coords)):
                coords = np.array([bbox[0], bbox[1]])
            y = min(int(coords[1] // self.sector_len[1]), HEIGHT_SECTORS-1)
            x = min(int(coords[0] // self.sector_len[0]), WIDTH_SECTORS-1)
#             print(y, x)
            if self.average_height_sectors[y, x][1] == 0:
                return np.nan
            height_sum, count = self.average_height_sectors[y, x]
            pixel_height = height_sum / count
            average_pixel_height = height_sum / count
        else:
            coords = coords.astype(np.int32)
            #update sector's average pixel height
            y = min(coords[1] // self.sector_len[1], HEIGHT_SECTORS-1)
            x = min(coords[0] // self.sector_len[0], WIDTH_SECTORS-1)
            self.average_height_sectors[y, x] += np.array([pixel_height, 1])

            height_sum, count = self.average_height_sectors[y, x]
            average_pixel_height = height_sum / count
        
        height = pixel_height / average_pixel_height * AVERAGE_HEIGHT

        pixel_per_meter = pixel_height / height
        if self.prev_coords is None or np.all(self.prev_coords == [0, 0]):
            self.prev_coords = coords
            return np.nan

        self.speed_data[self.cursor % SPEED_WINDOW] = (coords - self.prev_coords) / pixel_per_meter / (frame_id - self.frame_id)                                                             
        self.cursor += 1
        self.frame_id = frame_id
        self.prev_coords = coords

        return self.normal_speed()
    
    def vector_speed(self):
        speeds = self.speed_data[np.all(self.speed_data[:] != [0, 0], axis=1)]
        return np.sum(speeds, axis=0) / (self.period * speeds.shape[0] * FPS_DIVIDER)
    
    def normal_speed(self):
        speeds = np.abs(self.speed_data[np.all(self.speed_data[:] != [0, 0], axis=1)])
        result_speed = np.linalg.norm(np.sum(speeds, axis=0) / (self.period * speeds.shape[0] * FPS_DIVIDER))
        return result_speed if not np.isinf(result_speed) else 0

def get_pixel_height(kp):

    head = kp[HEAD_POINTS]
    top = head[np.all(head[:] != [0, 0], axis=1)].transpose((1, 0)).mean(axis=1)[1]

    # legs' pixel length  
    right = kp[RIGHT_HIP][1] + np.linalg.norm(kp[RIGHT_HIP] - kp[RIGHT_KNEE]) + np.linalg.norm(kp[RIGHT_KNEE] - kp[RIGHT_ANKLE])
    left = kp[LEFT_HIP][1] + np.linalg.norm(kp[LEFT_HIP] - kp[LEFT_KNEE]) + np.linalg.norm(kp[LEFT_KNEE] - kp[LEFT_ANKLE])

    # check if ankle and hip points exist
    right_state = not np.all(kp[RIGHT_HIP] == [0, 0]) and not np.all(kp[RIGHT_ANKLE] == [0, 0]) and not np.all(kp[RIGHT_KNEE] == [0, 0])
    left_state = not np.all(kp[LEFT_HIP] == [0, 0]) and not np.all(kp[LEFT_ANKLE] == [0, 0]) and not np.all(kp[LEFT_KNEE] == [0, 0])
    if right_state and left_state:
        return abs(top - max(right, left))
    elif right_state:
        return abs(top - right)
    elif left_state:
        return abs(top - left)
    return np.nan


def get_coords(kp):
    body = kp[TRACK_POINTS]
    return body[np.all(body[:] != [0, 0], axis=1)].transpose((1, 0)).mean(axis=1)

def get_chest(kp):
    body = kp[CHEST_POINTS]
    return body[np.all(body[:] != [0, 0], axis=1)].transpose((1, 0)).mean(axis=1)


class SpeedGraph:
    def __init__(self, start_frame_id):
        self.start_frame_id = start_frame_id
        self.graph = np.zeros((224, 224, 3))
        self.last_x, self.last_x_i, self.last_y, self.last_y_i, self.last_s, self.last_s_i, self.last_cx, self.last_cy = None, None, None, None, None, None, None, None
    
    def update(self, speed_data, frame_id):
        s = speed_data['speed']
        x, y = speed_data['vector']['delta']

        curr_i = int((frame_id - self.start_frame_id)/FRAMES_PROCESSING*224)
        if not np.isnan(s):
            s = 224 - int(s * 48 + 112)
            if self.last_s:
                cv2.line(self.graph, (self.last_s_i, self.last_s), (curr_i, s), (0, 0, 255), 2)
            self.last_s, self.last_s_i = s, curr_i

        if not np.isnan(x):
            x = 224 - int(x / 600 * 110 + 112)
            if self.last_x:
                cv2.line(self.graph, (self.last_x_i, self.last_x), (curr_i, x), (255, 0, 0), 2)
            self.last_x, self.last_x_i = x, curr_i

        if not np.isnan(y):
            y = 224 - int(y / 600 * 110 + 112)
            if self.last_y:
                cv2.line(self.graph, (self.last_y_i, self.last_y), (curr_i, y), (0, 255, 0), 2)
            self.last_y, self.last_y_i = y, curr_i
    
    def get_graph(self):
        return self.graph.astype(np.float32).transpose(2, 0, 1) / 255


class SpeedTracker:
    def __init__(self, height, width, period) -> None:
        self.sector_len = np.array([width // WIDTH_SECTORS, height // HEIGHT_SECTORS])
        self.period = period
        self.average_speed = {}
        self.average_height_sectors = np.zeros((HEIGHT_SECTORS, WIDTH_SECTORS, 2))
        # self.session = session
        self.speed_graphs = {}
    
    def speed(self, poses: dict, frame_id):
        res = {}
        for track_id, (bbox, kp) in poses.items():
            if track_id not in self.average_speed.keys():
                self.average_speed[track_id] = TrackData(frame_id, self.average_height_sectors, self.sector_len, self.period)
            if track_id not in self.speed_graphs.keys():
                self.speed_graphs[track_id] = SpeedGraph(frame_id)
            chest = get_chest(kp)
            if frame_id % FPS_DIVIDER == 0:
                res[track_id] = {
                    "speed": self.average_speed[track_id].update_speed(bbox, kp, frame_id),
                    "vector": {
                        "delta": self.average_speed[track_id].vector_speed() * get_pixel_height(kp),
                        "coords": np.array((bbox[0], bbox[1])) if np.any(np.isnan(chest)) else chest
                    }
                }
            else:
                res[track_id] = {
                    "speed": self.average_speed[track_id].normal_speed(),
                    "vector": {
                        "delta": self.average_speed[track_id].vector_speed() * get_pixel_height(kp),
                        "coords": np.array((bbox[0], bbox[1])) if np.any(np.isnan(chest)) else chest
                    }
                }
            self.speed_graphs[track_id].update(res[track_id], frame_id)
        return res
    
    # def classify(self):
    #     count = len(self.speed_graphs)
    #     speed_classes = {}
    #     if count != 0:
    #         batch = np.zeros((count, 3, 224, 224), dtype = np.float32)
    #         for i, (track_id, g) in enumerate(self.speed_graphs.items()):
    #             batch[i] = g.get_graph()
    #         output_names = [x.name for x in self.session.get_outputs()]
    #         dets = self.session.run(output_names, {self.session.get_inputs()[0].name: batch})[0]

            
    #         for k, det in zip(self.speed_graphs.keys(), dets):
    #             speed_classes[k] = det
        
    #     self.speed_graphs = {}
    #     return speed_classes