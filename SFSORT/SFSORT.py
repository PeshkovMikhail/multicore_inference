"""This module contains the SFSORT object detection tracker
implementation. By introducing a novel cost function called
the Bounding Box Similarity Index, this project eliminates
the Kalman Filter, leading to reduced computational requirements.

.. include:: ./docs/documentation.md
"""
import numpy as np

# It is best to use (lapjv)[https://github.com/gatagat/lap]
# since the time to solve the linear assignment problem
# is the shortest for this implementation
use_lap=True
try:
    import lap
except ImportError:
    from scipy.optimize import linear_sum_assignment
    use_lap=False


class DotAccess(dict):
    """
    Provides dot.notation access to dictionary attributes

    Parameters
    ----------
    dict : dictionary
        The dictionary to access with dot notation

    Examples
    --------
    >>> d = DotAccess({'a': 1, 'b': 2})
    >>> d.a
    1
    >>> d.b
    2
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TrackState:
    """
    Enumeration of possible states of a track

    Attributes
    ----------
    Active : int
        The track is active
    Lost_Central : int
        The track is lost in the central region
    Lost_Marginal : int
        The track is lost in the marginal region
    """
    Active = 0
    Lost_Central = 1
    Lost_Marginal = 2


class Track:
    """
    Handles basic track attributes and operations

    Parameters
    ----------
    bbox : array_like
        The bounding box of the track
    frame_id : int
        The frame ID of the track
    track_id : int
        The track ID
    cls_id : int
        The class ID of the track
    score : float
        The score of the track

    Attributes
    ----------
    track_id : int
        The track ID
    bbox : array_like
        The bounding box of the track
    cls_id : int
        The class ID of the track
    score : float
        The score of the track
    state : int
        The state of the track (active, lost central, or lost marginal)
    last_frame : int
        The last frame ID of the track

    Examples
    --------
    >>> track = Track([1, 2, 3, 4], 0, 0, 0, 0.5)
    >>> track.update([5, 6, 7, 8], 1, 1, 0.8)
    >>> track.bbox
    [5, 6, 7, 8]
    >>> track.score
    0.8
    """
    def __init__(self, bbox, frame_id, track_id, cls_id, score):
        self.track_id = track_id
        self.update(bbox, frame_id, cls_id, score)
        self.state = TrackState.Active

    def update(self, bbox, frame_id, cls_id, score):
        """Updates a matched track"""
        self.bbox = bbox
        self.cls_id = cls_id
        self.score = score
        self.state = TrackState.Active
        self.last_frame = frame_id


class SFSORT:
    """
    Multi-Object Tracking System

    Parameters
    ----------
    args : dictionary
        The arguments for the tracker

    Attributes
    ----------
    frame_no : int
        The current frame number
    id_counter : int
        The track ID counter
    active_tracks : list
        The list of active tracks
    lost_tracks : list
        The list of lost tracks

    Examples
    --------
    >>> tracker = SFSORT({'low_th': 0.5, 'match_th_second': 0.7})
    >>> tracks = tracker.update([[124, 112, 327, 450], [234, 56, 261, 563]],
                                [0.8, 0.9],
                                [0, 1])
    >>> bbox_list      = tracks[:, 0]
    >>> track_id_list  = tracks[:, 1]
    >>> cls_id_list    = tracks[:, 2]
    >>> scores_list    = tracks[:, 3]
    """

    def __init__(self, args):
        # Initialize tracker's arguments
        self.update_args(args)

        # Initialize the tracker
        self.frame_no = 0
        self.id_counter = 0
        self.active_tracks = []
        self.lost_tracks = []

    def update_args(self, args):
        """Updates tracker's arguments

        Parameters
        ----------
        args : dict
            Tracker parameters
        """
        args = DotAccess(args)
        # Register tracking arguments

        self.low_th = args.low_th
        self.match_th_second = args.match_th_second

        self.high_th = args.high_th
        self.match_th_first = args.match_th_first
        self.new_track_th = args.new_track_th

        if args.dynamic_tuning:
            self.cth = args.cth if args.cth else 0.7
            self.hthm = args.high_th_m if args.high_th_m else 0
            self.nthm = args.new_track_th_m if args.new_track_th_m else 0
            self.mthm = args.match_th_first_m if args.match_th_first_m else 0


        self.marginal_timeout = args.marginal_timeout
        self.central_timeout = args.central_timeout
        self.l_margin = args.horizontal_margin
        self.t_margin = args.vertical_margin
        self.r_margin = args.frame_width - args.horizontal_margin
        self.b_margin = args.frame_height - args.vertical_margin

    def update(self, boxes, scores, class_ids):
        """Updates tracker with new detections

        Parameters
        ----------
        boxes : array
            Bounding boxes coordinates
        scores : array
            Neural network confidence values
        class_ids: array
            Detected class for an object
        Returns
        -------
        result : array
            All of the inputs plus track IDs for actively tracked objects
        """
        # Adjust dynamic arguments
        count = len(scores[scores>self.cth])

        if count < 1:
          count = 1

        lnc = np.log10(count)
        hth = self.high_th - (self.hthm * lnc)
        nth = self.new_track_th + (self.nthm * lnc)
        mth = self.match_th_first - (self.mthm * lnc)

        # Increase frame number
        self.frame_no += 1

        # Variable: Active tracks in the next frame
        next_active_tracks = []

        # Remove long-time lost tracks
        for track in self.lost_tracks:
            if track.state == TrackState.Lost_Central:
                if self.frame_no - track.last_frame > self.central_timeout:
                    self.lost_tracks.remove(track)
                    del track
            else:
                if self.frame_no - track.last_frame > self.marginal_timeout:
                    self.lost_tracks.remove(track)
                    del track

        # Gather out all previous tracks
        track_pool = self.active_tracks + self.lost_tracks

        # Try to associate tracks with high score detections
        unmatched_tracks = np.array([])
        high_score = scores > hth
        if high_score.any():
            definite_boxes = boxes[high_score]
            definite_scores = scores[high_score]
            definite_classes = class_ids[high_score]
            if track_pool:
                cost = self.calculate_cost(track_pool, definite_boxes)
                matches, unmatched_tracks, unmatched_detections = self.linear_assignment(cost, mth)
                # Update/Activate matched tracks
                for track_idx, detection_idx in matches:
                    box = definite_boxes[detection_idx]
                    class_id = definite_classes[detection_idx]
                    score = definite_scores[detection_idx]
                    track = track_pool[track_idx]
                    track.update(box, self.frame_no, class_id, score)
                    next_active_tracks.append(track)
                    # Remove re-identified tracks from lost list
                    if track in self.lost_tracks:
                        self.lost_tracks.remove(track)
                # Identify eligible unmatched detections as new tracks
                for detection_idx in unmatched_detections:
                    if definite_scores[detection_idx] > nth:
                        box = definite_boxes[detection_idx]
                        class_id = definite_classes[detection_idx]
                        score = definite_scores[detection_idx]
                        track = Track(box, self.frame_no, self.id_counter, class_id, score)
                        next_active_tracks.append(track)
                        self.id_counter += 1
            else:
            	# Associate tracks of the first frame after object-free/null frames
                for detection_idx, score in enumerate(definite_scores):
                    if score > nth:
                        box = definite_boxes[detection_idx]
                        class_id = definite_classes[detection_idx]
                        score = definite_scores[detection_idx]
                        track = Track(box, self.frame_no, self.id_counter, class_id, score)
                        next_active_tracks.append(track)
                        self.id_counter += 1

        # Add unmatched tracks to the lost list
        unmatched_track_pool = []
        for track_address in unmatched_tracks:
            unmatched_track_pool.append(track_pool[track_address])
        next_lost_tracks = unmatched_track_pool.copy()

        # Try to associate remained tracks with intermediate score detections
        intermediate_score = np.logical_and((self.low_th < scores), (scores < hth))
        if intermediate_score.any():
            if len(unmatched_tracks):
                possible_boxes = boxes[intermediate_score]
                possible_class_ids = class_ids[intermediate_score]
                possible_scores = scores[intermediate_score]
                cost = self.calculate_cost(unmatched_track_pool, possible_boxes, iou_only=True)
                matches, unmatched_tracks, unmatched_detections = self.linear_assignment(cost, self.match_th_second)
                # Update/Activate matched tracks
                for track_idx, detection_idx in matches:
                    box = possible_boxes[detection_idx]
                    class_id = possible_class_ids[detection_idx]
                    score = possible_scores[detection_idx]
                    track = unmatched_track_pool[track_idx]
                    track.update(box, self.frame_no, class_id, score)
                    next_active_tracks.append(track)
                    # Remove re-identified tracks from lost list
                    if track in self.lost_tracks:
                        self.lost_tracks.remove(track)
                    next_lost_tracks.remove(track)

        # All tracks are lost if there are no detections!
        if not (high_score.any() or  intermediate_score.any()):
            next_lost_tracks = track_pool.copy()

        # Update the list of lost tracks
        for track in next_lost_tracks:
            if track not in self.lost_tracks:
                self.lost_tracks.append(track)
                u = track.bbox[0] + (track.bbox[2] - track.bbox[0]/2)
                v = track.bbox[1] + (track.bbox[3] - track.bbox[1]/2)
                if (self.l_margin < u < self.r_margin) and (self.t_margin < v < self.b_margin):
                    track.state = TrackState.Lost_Central
                else:
                    track.state = TrackState.Lost_Marginal

        # Update the list of active tracks
        self.active_tracks = next_active_tracks.copy()

        result = np.asarray([
            [x.bbox, x.track_id, x.cls_id, round(x.score, 2)]
            for x in next_active_tracks],
            dtype=object)

        return result

    @staticmethod
    def calculate_cost(tracks, boxes, iou_only=False):
        """
        Calculates the association cost based on IoU and box similarity

        Parameters
        ----------
        tracks : list
            The list of tracks
        boxes : array_like
            The list of bounding boxes
        iou_only : bool, optional
            Whether to calculate IoU only (default is False)

        Returns
        -------
        cost_matrix : array
            The association cost matrix
        """
        eps = 1e-7
        active_boxes = [track.bbox for track in tracks]

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = np.array(active_boxes).T
        b2_x1, b2_y1, b2_x2, b2_y2 = np.array(boxes).T

        h_intersection = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0)
        w_intersection = (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

        # Calculate the intersection area
        intersection =  h_intersection * w_intersection

        # Calculate the union area
        box1_height = b1_x2 - b1_x1
        box2_height = b2_x2 - b2_x1
        box1_width = b1_y2 - b1_y1
        box2_width = b2_y2 - b2_y1

        box1_area = box1_height * box1_width
        box2_area = box2_height * box2_width

        union = (box2_area + box1_area[:, None] - intersection + eps)

        # Calculate the IoU
        iou = intersection / union

        if iou_only:
            return 1.0 - iou

        # Calculate the DIoU
        centerx1 = (b1_x1 + b1_x2) / 2.0
        centery1 = (b1_y1 + b1_y2) / 2.0
        centerx2 = (b2_x1 + b2_x2) / 2.0
        centery2 = (b2_y1 + b2_y2) / 2.0
        inner_diag = np.abs(centerx1[:, None] - centerx2) + np.abs(centery1[:, None] - centery2)

        xxc1 = np.minimum(b1_x1[:, None], b2_x1)
        yyc1 = np.minimum(b1_y1[:, None], b2_y1)
        xxc2 = np.maximum(b1_x2[:, None], b2_x2)
        yyc2 = np.maximum(b1_y2[:, None], b2_y2)
        outer_diag = np.abs(xxc2 - xxc1) + np.abs(yyc2 - yyc1)

        diou = iou - (inner_diag / outer_diag)

        # Calculate the BBSI
        delta_w = np.abs(box2_width - box1_width[:, None])
        sw = w_intersection / np.abs(w_intersection + delta_w + eps)

        delta_h = np.abs(box2_height - box1_height[:, None])
        sh = h_intersection / np.abs(h_intersection + delta_h + eps)

        bbsi = diou + sh + sw

        # Normalize the BBSI
        cost = (bbsi)/3.0

        return 1.0 - cost


    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        """
        Linear assignment

        Parameters
        ----------
        cost_matrix : array
            The association cost matrix
        thresh : float
            The threshold for the linear assignment

        Returns
        -------
        matches : array
            The matched indices
        unmatched_tracks : tuple
            The unmatched track indices
        unmatched_detections : tuple
            The unmatched detection indices
        """
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

        if use_lap:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        else:
            y, x = linear_sum_assignment(cost_matrix)
            matches = np.asarray([[i, x] for i, x in enumerate(x) if cost_matrix[i, x] <= thresh])
            unmatched = np.ones(cost_matrix.shape)
            for i, xi in matches:
                unmatched[i, xi] = 0.0
            unmatched_a = np.where(unmatched.all(1))[0]
            unmatched_b = np.where(unmatched.all(0))[0]

        return matches, unmatched_a, unmatched_b
