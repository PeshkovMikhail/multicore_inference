import numpy as np
import cv2
from scipy.special import softmax

def letterbox(
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleup=True,
        stride=32,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """
        Resizes and pads an image to fit a new shape while maintaining aspect ratio.

        Parameters
        ----------
        im : np.ndarray
            The input image to be resized and padded.
        new_shape : tuple[int, int], optional
            The desired output shape (height, width). Default is (640, 640).
        color : tuple[int, int, int], optional
            The color for padding. Default is (114, 114, 114).
        auto : bool, optional
            If True, adjusts padding to be a multiple of stride. Default is True.
        scaleup : bool, optional
            If True, allows scaling up the image. If False, only scales down. Default is True.
        stride : int, optional
            The stride for padding adjustment. Default is 32.

        Returns
        -------
        tuple[np.ndarray, float, tuple[float, float]]
            A tuple containing the resized and padded image,
            the scaling ratio, and the padding values.
        """

        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, r, (dw, dh)

def dfl(position):
    # Distribution Focal Loss (DFL)
    x = np.array(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape((n,p_num,mc,h,w))
    y = softmax(y, 2)
    acc_metrix = np.arange(mc, dtype=np.float32).reshape((1,1,mc,1,1))
    y = np.sum(y*acc_metrix, 2)
    return y




class YOLOv10:
    def __init__(self, rknn_lite, net_size: int = 640):
        self.rknn_lite = rknn_lite
        self.net_size = net_size
        
        self.HeadNum = 3
        self.Strides = [8, 16, 32]
        self.MapSize = ((80, 80), (40, 40), (20, 20))

        self.NmsThresh = 0.45
        self.ObjectThresh = 0.25

        self.TopK = 50
        self.RegDfl = []
        self.RegDeq = [0]*16

        self.MeshGrid = []

        for index in range(self.HeadNum):
            for i in range(self.MapSize[index][0]):
                for j in range(self.MapSize[index][1]):
                    self.MeshGrid.append(j+0.5)
                    self.MeshGrid.append(i+0.5)


    def pre_process(self, img: np.ndarray) -> np.ndarray:
        img, ratio, dwdh = letterbox(img, auto=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        return img, ratio, dwdh

    def inference(self, img: np.ndarray) -> list[np.ndarray] | None:
        return self.rknn_lite.inference(inputs=[img])

    def post_process(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        gridIndex = -2
        RectResults = []
        for index in range(self.HeadNum):
            reg = outputs[index*2].flatten()
            cls = outputs[index*2 + 1]

            for h in range(self.MapSize[index][0]):
                for w in range(self.MapSize[index][1]):
                    gridIndex += 2
                    cls_max = sigmoid(cls[0, 0,h,w])
                    cls_index = 0
                    if cls_max > self.ObjectThresh:
                        self.RegDfl.clear()
                        for lc in range(4):
                            sfsum = 0
                            locval = 0
                            for df in range(16):
                                locvaltemp = np.exp(reg[((lc * 16) + df) * self.MapSize[index][0] * self.MapSize[index][1] + h * self.MapSize[index][1] + w])
                                self.RegDeq[df] = locvaltemp
                                sfsum += locvaltemp
                            for df in range(16):
                                locvaltemp = self.RegDeq[df] / sfsum
                                locval += locvaltemp * df
                            self.RegDfl.append(locval)
                        
                        xmin = (self.MeshGrid[gridIndex + 0] - self.RegDfl[0]) * self.Strides[index]
                        ymin = (self.MeshGrid[gridIndex + 1] - self.RegDfl[1]) * self.Strides[index]
                        xmax = (self.MeshGrid[gridIndex + 0] + self.RegDfl[2]) * self.Strides[index]
                        ymax = (self.MeshGrid[gridIndex + 1] + self.RegDfl[3]) * self.Strides[index]

                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(640, xmax)
                        ymax = min(640, ymax)

                        RectResults.append((xmin, ymin, xmax, ymax, cls_index, cls_max))
        if len(RectResults) > self.TopK:
            RectResults.sort(key= lambda x: x[5])
        return np.array(RectResults[:self.TopK])

    def draw(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = map(int, box)
            cv2.rectangle(
                img=img,
                pt1=(top, left),
                pt2=(right, bottom),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.putText(
                img=img,
                text=f"{self.classes[cl]} {score:.2f}",
                org=(top, left - 6),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 255),
                thickness=2,
            )

        return img

    def run(self, img: np.ndarray) -> np.ndarray:
        pre_img, ratio, dwdh = self.pre_process(img)
        outputs = self.inference(pre_img)
        if outputs is None:
            return img

        bboxes = self.post_process(outputs)
        bboxes[:, :4] -= np.array(dwdh*2)
        bboxes[:,:4] /= ratio
        return bboxes

        if all(x is not None for x in (boxes, classes, scores)):
            boxes -= np.array(dwdh * 2)
            boxes /= ratio
            boxes = boxes.round().astype(np.int32)

            inf_img = self.draw(img, boxes, classes, scores) # type: ignore

            return inf_img

        return img

class YOLOv8:
    def __init__(self, rknn_lite, net_size=(960, 960), obj_thresh = 0.25, nms_thresh = 0.45):
        self.net_size = net_size
        self.session = rknn_lite
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= self.obj_thresh)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.net_size[1]//grid_h, self.net_size[0]//grid_w]).reshape(1,2,1,1)

        position = dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy
    
    def post_process(self, input_data, dydx, ratio):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        inds = np.where(classes == 0)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = self.nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        boxes -= np.array(dydx*2)
        boxes /= ratio
        boxes = boxes.astype(np.int32)

        return np.concatenate([boxes, classes[:, None], scores[:, None]], axis=1)
    
    def run(self, img):
        feed, ratio, dydx = letterbox(img, (960, 960), auto=False)
        feed = np.expand_dims(feed, 0)
        results = self.session.inference(inputs=[feed])
        return self.post_process(results, dydx, ratio)
        person_ids = classes == 0
        bboxes = bboxes[person_ids]
        scores = scores[person_ids]

        
        return bboxes, classes[person_ids], scores