'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from scipy.optimize import linear_sum_assignment
from cv2 import KalmanFilter


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, detection, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.KF = self.init_kalmanfilter(detection)  # KF instance to track this object

        self.trace = []  # trace path

    @staticmethod
    def init_kalmanfilter(detection):
        KF = KalmanFilter(4, 2)
        dt = 1.0 / 30.0
        KF.transitionMatrix = np.array([[1, 0, dt, 0],
                                        [0, 1, 0, dt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]).astype(np.float32)
        KF.processNoiseCov = (np.diag([20, 20, 150, 150]).astype(np.float32) ** 2) * dt

        KF.statePost = np.array([[detection[0]],
                                 [detection[1]],
                                 [0],
                                 [0]]).astype(np.float32)
        KF.errorCovPost = np.diag([10, 10, 200, 200]).astype(np.float32) ** 2

        KF.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]]).astype(np.float32)
        KF.measurementNoiseCov = (np.eye(2).astype(np.float32) * 10) ** 2

        KF.statePre = KF.statePost  # maybe necessary?
        KF.errorCovPre = KF.errorCovPost
        return KF

    def predict(self):
        x = self.KF.predict()
        y = np.dot(self.KF.measurementMatrix, x)
        return y

    def correct(self, y):
        self.KF.correct(np.array(y).astype(np.float32))
        y = np.dot(self.KF.measurementMatrix, self.KF.statePost)
        return y

    def position_error(self):
        return np.sqrt(self.KF.errorCovPost[0, 0] + self.KF.errorCovPost[1, 1])

    def position(self):
        y = np.dot(self.KF.measurementMatrix, self.KF.statePost)
        return(y)


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_position_error, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_position_error = max_position_error
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i, track in enumerate(self.tracks):
            track.predict()  # Update KalmanFilter state

            for j, detection in enumerate(detections):
                diff = track.position() - np.array(detection).reshape(-1, 1)
                distance = np.sqrt(diff[0]**2 + diff[1]**2)
                cost[i][j] = distance

        # add costs of non assignment
        cost_non_assignment = self.dist_thresh / 2.0
        nont = np.ones((M, M)) * cost_non_assignment
        nond = np.ones((N, N)) * cost_non_assignment
        nonz = np.zeros((M, N))
        cost_aug = np.append(np.append(cost, nont, axis=0),
                             np.append(nond, nonz, axis=0), axis=1)

        row_ind, col_ind = linear_sum_assignment(cost_aug)

        for i, it in enumerate(row_ind):
            if col_ind[i] < M and it < N:
                self.tracks[it].correct(detections[col_ind[i]])

        # Start new tracks
        un_assigned_detects = np.intersect1d(col_ind[row_ind >= N], range(M))
        for id in un_assigned_detects:
            track = Track(detections[id], self.trackIdCount)
            self.trackIdCount += 1
            self.tracks.append(track)

        # Remove tracks undetected for too long
        del_tracks = []
        for i, track in enumerate(self.tracks):
            if (track.position_error() > self.max_position_error):
                del_tracks.append(i)

        for id in np.flipud(del_tracks):  # only when skipped frame exceeds max
            del self.tracks[id]

        # Update lastResults and tracks trace
        for track in self.tracks:
            if(len(track.trace) > self.max_trace_length):
                del track.trace[0]

            track.trace.append(track.position())
