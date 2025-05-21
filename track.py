import cv2
import mediapipe as mp
import numpy as np
import time
import platform # Added for OS detection
import logging # Added for logging
import math # Added for angle calculations
import sys # Added for PyQt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout # Added PyQt modules
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer # Added for UI updates

# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import pygrabber for camera names on Windows
try:
    from pygrabber.dshow_graph import FilterGraph
    PYGRABBER_AVAILABLE = True
except ImportError:
    PYGRABBER_AVAILABLE = False

class PoseTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        
        # For landmark names access
        self.lmk = self.mp_pose.PoseLandmark

        # Exercise tracking counters and states
        # Scissors exercise tracking (Arms)
        self.scissors_reps = 0
        self.scissors_arm_phase = None  # Can be 'LEFT_HIGH', 'RIGHT_HIGH'
        self.SCISSORS_THRESHOLD_RATIO = 0.15  # Relative to torso height for vertical separation

        # Head Tilt exercise tracking (side-to-side tilt)
        self.head_tilt_reps = 0
        self.head_tilt_phase = None # Can be 'LEFT', 'RIGHT', None (relative to shoulders Y-axis)
        self.HEAD_TILT_THRESHOLD_RATIO = 0.10 # Relative to shoulder distance for horizontal tilt

        # Head Rotation exercise tracking (side-to-side turn)
        self.head_rot_reps = 0
        self.head_rot_phase = None # Can be 'LEFT', 'RIGHT', None (relative to ears midpoint)
        self.HEAD_ROT_THRESHOLD_RATIO = 0.4 # Threshold for normalized horizontal nose position relative to ear midpoint

        # Bicep Curl exercise tracking
        self.bicep_curl_reps_left = 0
        self.bicep_curl_state_left = 'down' # Can be 'down', 'up'
        self.bicep_curl_reps_right = 0
        self.bicep_curl_state_right = 'down' # Can be 'down', 'up'
        self.BICEP_CURL_DOWN_ANGLE = 160 # Angle when arm is straight (or nearly straight)
        self.BICEP_CURL_UP_ANGLE = 50    # Angle when arm is curled (adjust based on observation)

        # Overhead Press exercise tracking (Both arms)
        self.overhead_press_reps = 0
        self.overhead_press_state = 'down' # Can be 'down', 'up'
        self.OVERHEAD_PRESS_THRESHOLD_RATIO = 0.2 # Wrist Y relative to shoulder Y threshold (relative to torso height)

        # Triceps Extension exercise tracking (Both arms)
        self.triceps_ext_reps = 0
        self.triceps_ext_state = 'down' # Can be 'down', 'up'
        self.TRICEPS_EXT_DOWN_ANGLE = 50 # Angle when arm is bent (adjust based on observation)
        self.TRICEPS_EXT_UP_ANGLE = 160  # Angle when arm is straight (or nearly straight)

        # Push-up exercise tracking
        self.pushup_reps = 0
        self.pushup_state = 'down' # Can be 'down', 'up'
        self.PUSHUP_DOWN_ANGLE = 90 # Angle at elbow when in the down position (adjust)
        self.PUSHUP_UP_ANGLE = 160  # Angle at elbow when in the up position (adjust)
        self.PUSHUP_VERTICAL_THRESHOLD_RATIO = 0.1 # Threshold for vertical position of shoulder relative to wrist (relative to torso height)

        # Row exercise tracking (Both arms)
        self.row_reps = 0
        self.row_state = 'down' # Can be 'down', 'up'
        self.ROW_ELBOW_ANGLE_THRESHOLD = 100 # Angle at elbow when pulled back (adjust)
        self.ROW_SHOULDER_ANGLE_THRESHOLD = 40 # Angle of upper arm relative to torso (adjust)

        # Arm Swings (Forward/Backward dynamic stretch)
        self.arm_swings_reps = 0
        self.arm_swings_state = 'down' # Can be 'down', 'up'
        self.ARM_SWINGS_UP_THRESHOLD_RATIO = 0.4 # Wrist Y significantly above shoulder Y (relative to shoulder distance)
        self.ARM_SWINGS_DOWN_THRESHOLD_RATIO = 0.4 # Wrist Y significantly below shoulder Y (relative to shoulder distance)

    def _get_landmark_coords(self, landmark_list, landmark_enum, image_width, image_height):
        """Safely gets landmark coordinates (0-1) and visibility if visible."""
        if landmark_list and len(landmark_list) > landmark_enum.value:
            lm = landmark_list[landmark_enum.value]
            if lm.visibility > 0.7:  # Increased visibility threshold for reliability
                return lm.x, lm.y, lm.visibility
        return None, None, None # Return None for all if not visible

    def _get_angle(self, p1_raw, p2_raw, p3_raw):
        """Calculates the angle in degrees between three points p1, p2, and p3 where p2 is the vertex.
           Uses raw (normalized) coordinates.
        """
        # Ensure points are not None
        if not all(coord is not None for coord in [*p1_raw[:2], *p2_raw[:2], *p3_raw[:2]]):
             return None

        # Extract x, y coordinates
        x1, y1 = p1_raw[:2]
        x2, y2 = p2_raw[:2]
        x3, y3 = p3_raw[:2]

        # Calculate angle using arctan2
        angle_radians = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle_degrees = math.degrees(angle_radians)

        # Ensure angle is positive (0 to 180 degrees)
        angle_degrees = abs(angle_degrees)
        if angle_degrees > 180.0:
            angle_degrees = 360 - angle_degrees

        return angle_degrees

    def _calculate_scissors_reps(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates using the helper (now returns raw + visibility)
        ls_x_raw, ls_y_raw, ls_vis = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        rs_x_raw, rs_y_raw, rs_vis = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)
        lh_x_raw, lh_y_raw, lh_vis = self._get_landmark_coords(landmark_list, self.lmk.LEFT_HIP, image_width, image_height)
        rh_x_raw, rh_y_raw, rh_vis = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_HIP, image_width, image_height)
        lw_x_raw, lw_y_raw, lw_vis = self._get_landmark_coords(landmark_list, self.lmk.LEFT_WRIST, image_width, image_height)
        rw_x_raw, rw_y_raw, rw_vis = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_WRIST, image_width, image_height)

        # Check if all crucial landmarks for this exercise are visible
        if not all(v is not None for v in [ls_vis, rs_vis, lh_vis, rh_vis, lw_vis, rw_vis]):
            # logging.debug("Scissors: Missing crucial landmarks or low visibility.")
            self.scissors_arm_phase = None # Reset phase if landmarks are lost
            return # Skip rep counting for this frame

        # Convert raw coordinates to pixel coordinates AFTER visibility check
        ls_y, rs_y, lh_y, rh_y, lw_y, rw_y = (
            ls_y_raw * image_height, rs_y_raw * image_height, 
            lh_y_raw * image_height, rh_y_raw * image_height, 
            lw_y_raw * image_height, rw_y_raw * image_height
        )

        # Calculate torso height (approx) for dynamic threshold
        avg_shoulder_y = (ls_y + rs_y) / 2
        avg_hip_y = (lh_y + rh_y) / 2
        torso_height_pixels = abs(avg_shoulder_y - avg_hip_y)

        if torso_height_pixels < 50: # Arbitrary minimum to prevent issues with tiny/invalid torso height
            # logging.debug(f"Scissors: Torso height ({torso_height_pixels:.2f}) too small or invalid.")
            self.scissors_arm_phase = None # Reset phase if torso height is invalid
            return

        vertical_separation_threshold = self.SCISSORS_THRESHOLD_RATIO * torso_height_pixels

        # Determine which arm is higher, if any, beyond the threshold
        current_high_arm = None
        if lw_y < rw_y - vertical_separation_threshold: # Y-axis is downwards, so smaller Y is higher
            current_high_arm = 'LEFT_HIGH'
        elif rw_y < lw_y - vertical_separation_threshold:
            current_high_arm = 'RIGHT_HIGH'
        
        # State machine for counting reps
        if current_high_arm:
            if self.scissors_arm_phase and self.scissors_arm_phase != current_high_arm:
                # A switch in which arm is high indicates a completed half-cycle by the previous high arm,
                # and the start of a new half-cycle by the current high arm.
                # We count a full rep when the phase changes (e.g. Left was high, now Right is high).
                self.scissors_reps += 1
                # logging.debug(f"Scissors Arm Rep! From {self.scissors_arm_phase} to {current_high_arm}. Total: {self.scissors_reps}")
            
            # Always update phase if a clear high arm is detected, even if it's the same as before.
            # This handles cases where arms might briefly enter the threshold zone.
            self.scissors_arm_phase = current_high_arm
        # If arms are too close (neither is significantly higher), maintain the last phase.
        # This prevents losing state if arms momentarily align or cross very slightly.

    def _calculate_head_tilts(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates using the helper (now returns raw + visibility)
        nose_x_raw, nose_y_raw, nose_vis = self._get_landmark_coords(landmark_list, self.lmk.NOSE, image_width, image_height)
        ls_x_raw, ls_y_raw, ls_vis = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        rs_x_raw, rs_y_raw, rs_vis = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)

        # Check if crucial landmarks are visible
        if not all(v is not None for v in [nose_vis, ls_vis, rs_vis]):
            # logging.debug("Head Tilts: Missing crucial landmarks or low visibility.")
            self.head_tilt_phase = None # Reset phase if landmarks are lost
            return # Skip rep counting

        # Convert raw coordinates to pixel coordinates AFTER visibility check
        nose_x = nose_x_raw * image_width
        ls_x = ls_x_raw * image_width
        rs_x = rs_x_raw * image_width

        # Calculate horizontal midpoint and distance between shoulders
        shoulder_mid_x = (ls_x + rs_x) / 2
        shoulder_distance_pixels = abs(ls_x - rs_x)

        if shoulder_distance_pixels < 50: # Arbitrary minimum distance
             # logging.debug(f"Head Tilts: Shoulder distance ({shoulder_distance_pixels:.2f}) too small or invalid.")
             self.head_tilt_phase = None # Reset phase if shoulder distance is invalid
             return

        horizontal_separation_threshold = self.HEAD_TILT_THRESHOLD_RATIO * shoulder_distance_pixels

        # Determine current head tilt phase
        current_tilt_phase = None
        if nose_x < shoulder_mid_x - horizontal_separation_threshold: # Nose is significantly left of center
            current_tilt_phase = 'LEFT'
        elif nose_x > shoulder_mid_x + horizontal_separation_threshold: # Nose is significantly right of center
            current_tilt_phase = 'RIGHT'
        # If neither, current_tilt_phase remains None (center or slightly tilted)

        # State machine for counting reps
        if current_tilt_phase:
            if self.head_tilt_phase and self.head_tilt_phase != current_tilt_phase:
                # Changed from LEFT to RIGHT or RIGHT to LEFT = 1 rep
                self.head_tilt_reps += 1
                # logging.debug(f"Head Tilt Rep! From {self.head_tilt_phase} to {current_tilt_phase}. Total: {self.head_tilt_reps}")

            # Update phase if a clear tilt is detected
            self.head_tilt_phase = current_tilt_phase
        # If currently centered, maintain the last tilt phase. This allows completing a rep
        # by returning to center and then tilting the other way.
        # The rep is counted on the transition from one side to the other.

    def _calculate_head_rotations(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates (raw + visibility)
        nose_x_raw, nose_y_raw, nose_vis = self._get_landmark_coords(landmark_list, self.lmk.NOSE, image_width, image_height)
        le_x_raw, le_y_raw, le_vis = self._get_landmark_coords(landmark_list, self.lmk.LEFT_EAR, image_width, image_height)
        re_x_raw, re_y_raw, re_vis = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_EAR, image_width, image_height)

        # Check if crucial landmarks are visible
        if not all(v is not None for v in [nose_vis, le_vis, re_vis]):
            # logging.debug("Head Rotations: Missing crucial landmarks or low visibility (Nose, Ears).")
            self.head_rot_phase = None # Reset phase if landmarks are lost
            return # Skip rep counting

        # Calculate horizontal midpoint and distance between ears (using raw coordinates)
        ear_mid_x_raw = (le_x_raw + re_x_raw) / 2
        ear_distance_raw = abs(le_x_raw - re_x_raw)

        if ear_distance_raw < 0.05: # Arbitrary minimum normalized distance to avoid division by zero or tiny distances
             # logging.debug(f"Head Rotations: Ear distance ({ear_distance_raw:.2f}) too small or invalid.")
             self.head_rot_phase = None # Reset phase if ear distance is invalid
             return

        # Calculate normalized horizontal position of the nose relative to the ear midpoint
        # Value will be negative when nose is left of midpoint, positive when right.
        # Range approximately -0.5 to 0.5 (over ear span)
        normalized_nose_x_relative_to_ears = (nose_x_raw - ear_mid_x_raw) / ear_distance_raw

        # Determine current head rotation phase
        current_rot_phase = None
        if normalized_nose_x_relative_to_ears < -self.HEAD_ROT_THRESHOLD_RATIO: # Nose is significantly left of ear midpoint
            current_rot_phase = 'LEFT'
        elif normalized_nose_x_relative_to_ears > self.HEAD_ROT_THRESHOLD_RATIO: # Nose is significantly right of ear midpoint
            current_rot_phase = 'RIGHT'
        # If neither, current_rot_phase remains None (looking forward or slight turn)

        # State machine for counting reps
        if current_rot_phase:
            if self.head_rot_phase and self.head_rot_phase != current_rot_phase:
                # Changed from LEFT to RIGHT or RIGHT to LEFT = 1 rep
                self.head_rot_reps += 1
                # logging.debug(f"Head Rotation Rep! From {self.head_rot_phase} to {current_rot_phase}. Total: {self.head_rot_reps}")

            # Update phase if a clear rotation is detected
            self.head_rot_phase = current_rot_phase
        # If currently centered, maintain the last rotation phase. This allows completing a rep
        # by returning to center and then rotating the other way.
        # The rep is counted on the transition from one side to the other.

    def _calculate_bicep_curls(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates (raw + visibility)
        ls_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        le_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_ELBOW, image_width, image_height)
        lw_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_WRIST, image_width, image_height)
        rs_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)
        re_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_ELBOW, image_width, image_height)
        rw_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_WRIST, image_width, image_height)

        # --- Left Bicep Curl ---
        # Check if crucial landmarks for left arm are visible
        if all(v is not None for v in [ls_raw[2], le_raw[2], lw_raw[2]]):
            left_elbow_angle = self._get_angle(ls_raw, le_raw, lw_raw)
            if left_elbow_angle is not None:
                # logging.debug(f"Left Elbow Angle: {left_elbow_angle:.2f}")
                # State machine
                if left_elbow_angle > self.BICEP_CURL_DOWN_ANGLE:
                    current_state = 'down'
                elif left_elbow_angle < self.BICEP_CURL_UP_ANGLE:
                    current_state = 'up'
                else:
                    current_state = self.bicep_curl_state_left # Maintain state in transition range

                if current_state == 'down' and self.bicep_curl_state_left == 'up':
                    self.bicep_curl_reps_left += 1
                    # logging.debug(f"Left Bicep Curl Rep! Total: {self.bicep_curl_reps_left}")

                self.bicep_curl_state_left = current_state
        else:
             self.bicep_curl_state_left = 'down' # Reset state if landmarks are lost

        # --- Right Bicep Curl ---
        # Check if crucial landmarks for right arm are visible
        if all(v is not None for v in [rs_raw[2], re_raw[2], rw_raw[2]]):
            right_elbow_angle = self._get_angle(rs_raw, re_raw, rw_raw)
            if right_elbow_angle is not None:
                # logging.debug(f"Right Elbow Angle: {right_elbow_angle:.2f}")
                # State machine
                if right_elbow_angle > self.BICEP_CURL_DOWN_ANGLE:
                    current_state = 'down'
                elif right_elbow_angle < self.BICEP_CURL_UP_ANGLE:
                    current_state = 'up'
                else:
                    current_state = self.bicep_curl_state_right # Maintain state in transition range

                if current_state == 'down' and self.bicep_curl_state_right == 'up':
                    self.bicep_curl_reps_right += 1
                    # logging.debug(f"Right Bicep Curl Rep! Total: {self.bicep_curl_reps_right}")

                self.bicep_curl_state_right = current_state
        else:
             self.bicep_curl_state_right = 'down' # Reset state if landmarks are lost

    def _calculate_overhead_press(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates (raw + visibility)
        ls_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        rs_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)
        lw_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_WRIST, image_width, image_height)
        rw_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_WRIST, image_width, image_height)
        lh_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_HIP, image_width, image_height) # Needed for torso height
        rh_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_HIP, image_width, image_height) # Needed for torso height

        # Check if crucial landmarks are visible
        if not all(v is not None for v in [ls_raw[2], rs_raw[2], lw_raw[2], rw_raw[2], lh_raw[2], rh_raw[2]]):
            # logging.debug("Overhead Press: Missing crucial landmarks or low visibility.")
            self.overhead_press_state = 'down' # Reset state if landmarks are lost
            return # Skip rep counting

        # Calculate average shoulder and wrist Y positions (raw/normalized)
        avg_shoulder_y_raw = (ls_raw[1] + rs_raw[1]) / 2
        avg_wrist_y_raw = (lw_raw[1] + rw_raw[1]) / 2

        # Calculate torso height (approx) using raw coordinates for dynamic threshold
        avg_hip_y_raw = (lh_raw[1] + rh_raw[1]) / 2
        torso_height_raw = abs(avg_shoulder_y_raw - avg_hip_y_raw)

        if torso_height_raw < 0.1: # Arbitrary minimum normalized height
             # logging.debug(f"Overhead Press: Torso height ({torso_height_raw:.2f}) too small or invalid.")
             self.overhead_press_state = 'down' # Reset state if torso height is invalid
             return

        # Calculate vertical separation threshold relative to torso height
        vertical_separation_threshold_raw = self.OVERHEAD_PRESS_THRESHOLD_RATIO * torso_height_raw

        # Determine current state based on wrist position relative to shoulder (Y-axis is downwards)
        # Arms are 'up' if wrists are significantly above shoulders.
        current_state = self.overhead_press_state # Maintain state by default
        if avg_wrist_y_raw < avg_shoulder_y_raw - vertical_separation_threshold_raw:
            current_state = 'up'
        elif avg_wrist_y_raw > avg_shoulder_y_raw + vertical_separation_threshold_raw:
             current_state = 'down'

        # State machine for counting reps
        if current_state == 'down' and self.overhead_press_state == 'up':
            self.overhead_press_reps += 1
            # logging.debug(f"Overhead Press Rep! Total: {self.overhead_press_reps}")

        self.overhead_press_state = current_state

    def _calculate_triceps_extensions(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates (raw + visibility)
        ls_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        le_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_ELBOW, image_width, image_height)
        lw_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_WRIST, image_width, image_height)
        rs_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)
        re_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_ELBOW, image_width, image_height)
        rw_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_WRIST, image_width, image_height)

        # Check if crucial landmarks for both arms are visible
        if not all(v is not None for v in [ls_raw[2], le_raw[2], lw_raw[2], rs_raw[2], re_raw[2], rw_raw[2]]):
            # logging.debug("Triceps Extensions: Missing crucial landmarks or low visibility.")
            self.triceps_ext_state = 'down' # Reset state if landmarks are lost
            return # Skip rep counting

        # Calculate elbow angles
        left_elbow_angle = self._get_angle(ls_raw, le_raw, lw_raw)
        right_elbow_angle = self._get_angle(rs_raw, re_raw, rw_raw)

        if left_elbow_angle is None or right_elbow_angle is None:
             self.triceps_ext_state = 'down' # Reset state if angle calculation failed
             return

        # Determine combined state (both arms should extend)
        current_state = self.triceps_ext_state # Maintain state by default
        if left_elbow_angle > self.TRICEPS_EXT_UP_ANGLE and right_elbow_angle > self.TRICEPS_EXT_UP_ANGLE:
            current_state = 'up'
        elif left_elbow_angle < self.TRICEPS_EXT_DOWN_ANGLE and right_elbow_angle < self.TRICEPS_EXT_DOWN_ANGLE:
            current_state = 'down'

        # State machine for counting reps
        if current_state == 'down' and self.triceps_ext_state == 'up':
            self.triceps_ext_reps += 1
            # logging.debug(f"Triceps Extension Rep! Total: {self.triceps_ext_reps}")

        self.triceps_ext_state = current_state

    def _calculate_pushups(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates (raw + visibility)
        ls_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        rs_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)
        le_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_ELBOW, image_width, image_height)
        re_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_ELBOW, image_width, image_height)
        lw_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_WRIST, image_width, image_height)
        rw_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_WRIST, image_width, image_height)
        lh_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_HIP, image_width, image_height) # Needed for torso height reference
        rh_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_HIP, image_width, image_height) # Needed for torso height reference

        # Check if crucial landmarks are visible
        if not all(v is not None for v in [ls_raw[2], rs_raw[2], le_raw[2], re_raw[2], lw_raw[2], rw_raw[2], lh_raw[2], rh_raw[2]]):
            # logging.debug("Pushups: Missing crucial landmarks or low visibility.")
            self.pushup_state = 'down' # Reset state if landmarks are lost
            return # Skip rep counting

        # Calculate average elbow angle
        left_elbow_angle = self._get_angle(ls_raw, le_raw, lw_raw)
        right_elbow_angle = self._get_angle(rs_raw, re_raw, rw_raw)

        if left_elbow_angle is None or right_elbow_angle is None:
             self.pushup_state = 'down' # Reset state if angle calculation failed
             return

        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

        # Calculate average shoulder and wrist Y positions (raw/normalized)
        avg_shoulder_y_raw = (ls_raw[1] + rs_raw[1]) / 2
        avg_wrist_y_raw = (lw_raw[1] + rw_raw[1]) / 2

         # Calculate torso height (approx) using raw coordinates for dynamic threshold
        avg_hip_y_raw = (lh_raw[1] + rh_raw[1]) / 2
        torso_height_raw = abs(avg_shoulder_y_raw - avg_hip_y_raw)

        if torso_height_raw < 0.1: # Arbitrary minimum normalized height
             # logging.debug(f"Pushups: Torso height ({torso_height_raw:.2f}) too small or invalid.")
             self.pushup_state = 'down' # Reset state if torso height is invalid
             return

        # Calculate vertical threshold based on torso height
        vertical_threshold_raw = self.PUSHUP_VERTICAL_THRESHOLD_RATIO * torso_height_raw

        # Determine current state based on elbow angle AND vertical position
        current_state = self.pushup_state # Maintain state by default
        if avg_elbow_angle > self.PUSHUP_UP_ANGLE and avg_shoulder_y_raw < avg_wrist_y_raw - vertical_threshold_raw: # Arms straighter AND shoulders above wrists
            current_state = 'up'
        elif avg_elbow_angle < self.PUSHUP_DOWN_ANGLE and avg_shoulder_y_raw > avg_wrist_y_raw + vertical_threshold_raw: # Arms bent AND shoulders below wrists
             current_state = 'down'

        # State machine for counting reps
        if current_state == 'up' and self.pushup_state == 'down': # Count a rep on the transition from down to up
            self.pushup_reps += 1
            # logging.debug(f"Pushup Rep! Total: {self.pushup_reps}")

        self.pushup_state = current_state

    def _calculate_rows(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates (raw + visibility)
        ls_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        le_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_ELBOW, image_width, image_height)
        lw_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_WRIST, image_width, image_height)
        rs_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)
        re_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_ELBOW, image_width, image_height)
        rw_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_WRIST, image_width, image_height)
        lh_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_HIP, image_width, image_height) # Needed for torso angle
        rh_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_HIP, image_width, image_height) # Needed for torso angle

        # Check if crucial landmarks are visible
        if not all(v is not None for v in [ls_raw[2], le_raw[2], lw_raw[2], rs_raw[2], re_raw[2], rw_raw[2], lh_raw[2], rh_raw[2]]):
            # logging.debug("Rows: Missing crucial landmarks or low visibility.")
            self.row_state = 'down' # Reset state if landmarks are lost
            return # Skip rep counting

        # Calculate elbow angles
        left_elbow_angle = self._get_angle(ls_raw, le_raw, lw_raw)
        right_elbow_angle = self._get_angle(rs_raw, re_raw, rw_raw)

        # Calculate torso angle (approximate, relative to vertical)
        avg_shoulder_raw = ((ls_raw[0] + rs_raw[0])/2, (ls_raw[1] + rs_raw[1])/2, min(ls_raw[2], rs_raw[2]))
        avg_hip_raw = ((lh_raw[0] + rh_raw[0])/2, (lh_raw[1] + rh_raw[1])/2, min(lh_raw[2], rh_raw[2]))
        # Create a point directly below shoulder to get vertical angle
        vertical_ref_shoulder = (avg_shoulder_raw[0], avg_shoulder_raw[1] + 0.1, avg_shoulder_raw[2]) # 0.1 is arbitrary vertical offset

        torso_angle_vertical = self._get_angle(avg_hip_raw, avg_shoulder_raw, vertical_ref_shoulder)

        if left_elbow_angle is None or right_elbow_angle is None or torso_angle_vertical is None:
             self.row_state = 'down' # Reset state if angle calculation failed
             return

        # Determine combined state (both arms should row simultaneously)
        current_state = self.row_state # Maintain state by default

        # Arms are 'up' when elbows are bent (rowing motion) AND torso is bent forward (typical row stance)
        # Arms are 'down' when elbows are straighter
        if (left_elbow_angle < self.ROW_ELBOW_ANGLE_THRESHOLD and right_elbow_angle < self.ROW_ELBOW_ANGLE_THRESHOLD):
             # Check if torso is bent forward enough (e.g., angle > 45 degrees from vertical)
             # This part is tricky and might need adjustment based on user's form
             if torso_angle_vertical > self.ROW_SHOULDER_ANGLE_THRESHOLD: # Shoulder ahead of hip horizontally
                  current_state = 'up'

        # Arms are 'down' if elbows are straighter (relaxed)
        if (left_elbow_angle > self.ROW_ELBOW_ANGLE_THRESHOLD + 20 and right_elbow_angle > self.ROW_ELBOW_ANGLE_THRESHOLD + 20): # +20 for a buffer
             current_state = 'down'

        # State machine for counting reps
        if current_state == 'up' and self.row_state == 'down': # Count a rep on the transition from down to up
            self.row_reps += 1
            # logging.debug(f"Row Rep! Total: {self.row_reps}")

        self.row_state = current_state

    def _calculate_arm_swings(self, results, image_width, image_height):
        if not results or not results.pose_landmarks:
            return

        landmark_list = results.pose_landmarks.landmark

        # Get required landmark coordinates (raw + visibility)
        ls_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_SHOULDER, image_width, image_height)
        rs_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_SHOULDER, image_width, image_height)
        lw_raw = self._get_landmark_coords(landmark_list, self.lmk.LEFT_WRIST, image_width, image_height)
        rw_raw = self._get_landmark_coords(landmark_list, self.lmk.RIGHT_WRIST, image_width, image_height)

        # Check if crucial landmarks are visible
        if not all(v is not None for v in [ls_raw[2], rs_raw[2], lw_raw[2], rw_raw[2]]):
            # logging.debug("Arm Swings: Missing crucial landmarks or low visibility.")
            self.arm_swings_state = 'down' # Reset state if landmarks are lost
            return # Skip rep counting

        # Calculate average shoulder and wrist Y positions (raw/normalized)
        avg_shoulder_y_raw = (ls_raw[1] + rs_raw[1]) / 2
        avg_wrist_y_raw = (lw_raw[1] + rw_raw[1]) / 2
        avg_shoulder_x_raw = (ls_raw[0] + rs_raw[0]) / 2 # Also get average shoulder X for threshold scaling
        shoulder_distance_raw = abs(ls_raw[0] - rs_raw[0])

        if shoulder_distance_raw < 0.05: # Arbitrary minimum normalized distance
             # logging.debug(f"Arm Swings: Shoulder distance ({shoulder_distance_raw:.2f}) too small or invalid.")
             self.arm_swings_state = 'down' # Reset state if shoulder distance is invalid
             return

        # Calculate vertical thresholds relative to shoulder distance (as a proxy for body scale)
        vertical_up_threshold_raw = self.ARM_SWINGS_UP_THRESHOLD_RATIO * shoulder_distance_raw
        vertical_down_threshold_raw = self.ARM_SWINGS_DOWN_THRESHOLD_RATIO * shoulder_distance_raw

        # Determine current state based on wrist position relative to average shoulder Y (Y-axis is downwards)
        current_state = self.arm_swings_state # Maintain state by default

        # Arms are 'up' if average wrist Y is significantly above average shoulder Y
        if avg_wrist_y_raw < avg_shoulder_y_raw - vertical_up_threshold_raw:
            current_state = 'up'
        # Arms are 'down' if average wrist Y is significantly below average shoulder Y
        elif avg_wrist_y_raw > avg_shoulder_y_raw + vertical_down_threshold_raw:
            current_state = 'down'

        # State machine for counting reps: Count a rep on the transition from 'up' to 'down'
        if current_state == 'down' and self.arm_swings_state == 'up':
            self.arm_swings_reps += 1
            # logging.debug(f"Arm Swing Rep! Total: {self.arm_swings_reps}")

        self.arm_swings_state = current_state

    def process_frame(self, frame, fps, actual_fps):
        """Processes a single frame to detect poses, count reps, and annotate the image."""
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape # Get frame dimensions here for calculations

        # Process the image and detect poses
        results = self.pose.process(image_rgb)

        # Draw pose landmarks on the image
        annotated_image = frame.copy()
        if results.pose_landmarks:
            # Define indices for upper body and face landmarks
            upper_body_landmarks = [
                self.lmk.NOSE.value,
                self.lmk.LEFT_EYE_INNER.value, self.lmk.LEFT_EYE.value, self.lmk.LEFT_EYE_OUTER.value,
                self.lmk.RIGHT_EYE_INNER.value, self.lmk.RIGHT_EYE.value, self.lmk.RIGHT_EYE_OUTER.value,
                self.lmk.LEFT_EAR.value, self.lmk.RIGHT_EAR.value,
                self.lmk.MOUTH_LEFT.value, self.lmk.MOUTH_RIGHT.value,
                self.lmk.LEFT_SHOULDER.value, self.lmk.RIGHT_SHOULDER.value,
                self.lmk.LEFT_ELBOW.value, self.lmk.RIGHT_ELBOW.value,
                self.lmk.LEFT_WRIST.value, self.lmk.RIGHT_WRIST.value,
                self.lmk.LEFT_PINKY.value, self.lmk.RIGHT_PINKY.value,
                self.lmk.LEFT_INDEX.value, self.lmk.RIGHT_INDEX.value,
                self.lmk.LEFT_THUMB.value, self.lmk.RIGHT_THUMB.value,
                # Include nose again for potential nose-to-face connections if needed (already in list, but ensures it's considered)
            ]
            # Add torso/hip landmarks which are often needed for calculations but exclude lower leg
            torso_hip_landmarks = [
                 self.lmk.LEFT_HIP.value,
                 self.lmk.RIGHT_HIP.value,
                 self.lmk.LEFT_SHOULDER.value, # Include shoulders again for clarity (already in list)
                 self.lmk.RIGHT_SHOULDER.value # Include shoulders again for clarity (already in list)
            ]
            # Combine and get unique indices, ensuring they are within bounds
            all_upper_body_indices = list(set(upper_body_landmarks + torso_hip_landmarks))
            all_upper_body_indices = [idx for idx in all_upper_body_indices if idx < len(results.pose_landmarks.landmark)]

            # Create a new landmark list containing only upper body landmarks
            upper_body_landmark_list = type(results.pose_landmarks)() # Create a new PoseLandmark object
            for i in all_upper_body_indices:
                 landmark = results.pose_landmarks.landmark[i]
                 # Need to copy the landmark data to the new list, maintaining the original structure
                 # This requires adding the landmark to the repeated field 'landmark' of the new object
                 new_landmark = upper_body_landmark_list.landmark.add()
                 new_landmark.CopyFrom(landmark)

            # Define connections for the upper body
            # This is a simplified approach - we'll include connections that primarily involve upper body landmarks
            upper_body_connections = []
            # Iterate through original connections and add if both points are in the upper body set
            upper_body_landmark_set = set(all_upper_body_indices)
            for connection in self.mp_pose.POSE_CONNECTIONS:
                 start_idx, end_idx = connection
                 # Ensure indices are within the bounds of the (original) landmark list
                 if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
                      if start_idx in upper_body_landmark_set and end_idx in upper_body_landmark_set:
                           upper_body_connections.append(connection)

            # Draw landmarks and connections for the upper body only
            self.mp_drawing.draw_landmarks(
                annotated_image,
                upper_body_landmark_list, # Use the filtered landmark list
                upper_body_connections,   # Use the filtered connections
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()) # Keep default style for drawn landmarks/connections

            # Calculate exercise reps if landmarks are available (use original results for calculations)
            self._calculate_scissors_reps(results, image_width, image_height)
            self._calculate_head_tilts(results, image_width, image_height)
            self._calculate_head_rotations(results, image_width, image_height)
            self._calculate_bicep_curls(results, image_width, image_height)
            self._calculate_overhead_press(results, image_width, image_height)
            self._calculate_triceps_extensions(results, image_width, image_height)
            self._calculate_pushups(results, image_width, image_height)
            self._calculate_rows(results, image_width, image_height)
            self._calculate_arm_swings(results, image_width, image_height)

        return annotated_image, results

    def close(self):
        self.pose.close()


def get_camera_names():
    """Returns a list of camera names if pygrabber is available, otherwise None."""
    if platform.system() == "Windows" and PYGRABBER_AVAILABLE:
        graph = FilterGraph()
        try:
            names = graph.get_input_devices() # Get list of available cameras
            return names
        except Exception as e:
            print(f"Error getting camera names with pygrabber: {e}")
            return None
    return None

def select_camera():
    """Lists available cameras and prompts the user to select one."""
    logging.debug("Attempting to select camera.")
    camera_names = get_camera_names()
    if camera_names:
        logging.debug(f"Camera names from pygrabber: {camera_names}")
    else:
        logging.debug("pygrabber not available or failed to get camera names. Will use indices.")
    
    index = 0
    arr = []
    display_names = []

    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
        if not cap.isOpened():
            logging.warning(f"Could not open camera index {index} to check if it exists.")
        if not cap.read()[0]:
            cap.release()
            break
        else:
            arr.append(index)
            if camera_names and index < len(camera_names):
                display_name = f"{index}: {camera_names[index]}"
                display_names.append(display_name)
                logging.debug(f"Found camera: {display_name}")
            else:
                display_name = f"{index}: Camera {index}"
                display_names.append(display_name)
                logging.debug(f"Found camera: {display_name}")
        cap.release()
        index += 1
    
    if not arr:
        print("Error: No cameras found.")
        logging.error("No cameras found during selection process.")
        return -1

    if len(arr) == 1:
        print(f"Defaulting to the only available camera: {display_names[0]}")
        logging.info(f"Only one camera found. Defaulting to: {display_names[0]} (index {arr[0]})")
        return arr[0]

    print("Available cameras:")
    for name in display_names:
        print(f"  {name}")

    while True:
        try:
            choice_str = input(f"Select camera by number (0-{len(arr)-1}): ")
            if not choice_str: # Handle empty input
                 print("No input detected. Please enter a number.")
                 logging.warning("Empty input for camera selection.")
                 continue
            choice = int(choice_str)
            if choice in arr:
                logging.info(f"User selected camera index: {choice}")
                return choice
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(arr)-1}.")
                logging.warning(f"Invalid camera choice: {choice}. Available: {arr}")
        except ValueError:
            print("Invalid input. Please enter a number.")
            logging.error("ValueError during camera selection input.")
        except Exception as e: # Catch other potential errors during input
            print(f"An unexpected error occurred: {e}")
            logging.error(f"Unexpected error during camera selection: {e}")

class VideoStreamWidget(QWidget):
    def __init__(self, tracker):
        super().__init__()

        self.tracker = tracker
        self.cap = None # OpenCV VideoCapture object
        self.actual_fps_from_cap = 0 # Store actual FPS reported by camera
        self.current_fps = 0 # Calculated processing FPS
        self.prev_time = 0 # For FPS calculation

        # QLabel for the video feed
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480) # Set a minimum size

        # Labels for displaying text information
        self.fps_label = QLabel("FPS: 0 (0 cap)")
        self.scissors_label = QLabel("Scissors Reps: 0")
        self.head_tilt_label = QLabel("Head Tilt Reps: 0")
        self.head_rot_label = QLabel("Head Rot Reps: 0")
        self.bicep_left_label = QLabel("Bicep Left: 0")
        self.bicep_right_label = QLabel("Bicep Right: 0")
        self.overhead_press_label = QLabel("Overhead Press: 0")
        self.triceps_ext_label = QLabel("Triceps Ext: 0")
        self.pushups_label = QLabel("Pushups: 0")
        self.rows_label = QLabel("Rows: 0")

        # Layout for text labels (vertical)
        text_layout = QVBoxLayout()
        text_layout.addWidget(self.fps_label)
        text_layout.addWidget(self.scissors_label)
        text_layout.addWidget(self.head_tilt_label)
        text_layout.addWidget(self.head_rot_label)
        text_layout.addWidget(self.bicep_left_label)
        text_layout.addWidget(self.bicep_right_label)
        text_layout.addWidget(self.overhead_press_label)
        text_layout.addWidget(self.triceps_ext_label)
        text_layout.addWidget(self.pushups_label)
        text_layout.addWidget(self.rows_label)
        text_layout.addStretch() # Add stretch to push labels to the top

        # Main layout (horizontal) - video feed on left, text labels on right
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, 1) # Stretch factor 1 for video feed
        main_layout.addLayout(text_layout, 0) # Stretch factor 0 for text labels

        self.setLayout(main_layout)

        # Timer to trigger frame reading and UI update
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_tracking(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

        # Attempt to set desired camera properties
        # Let's try a smaller resolution like 720p or 480p to improve FPS with MediaPipe
        desired_width = 1280  # Changed back to 1280 for 720p
        desired_height = 720   # Changed back to 720 for 720p
        # Or for even more performance, try 640x480
        # desired_width = 640
        # desired_height = 480
        desired_fps = 60

        logging.debug(f"Attempting to set camera properties: Width={desired_width}, Height={desired_height}, FPS={desired_fps}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        self.cap.set(cv2.CAP_PROP_FPS, desired_fps)

        # Check actual properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.actual_fps_from_cap = self.cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Requested: {desired_width}x{desired_height} @ {desired_fps}FPS")
        logging.info(f"Actual camera properties: Width={actual_width}, Height={actual_height}, FPS={self.actual_fps_from_cap}")

        if not self.cap.isOpened():
            logging.error(f"Failed to open camera with index {camera_index} after attempting to set properties.")
            return False
        
        logging.info(f"Successfully opened camera {camera_index} with reported properties: {actual_width}x{actual_height} @ {self.actual_fps_from_cap}FPS")

        self.timer.start(1) # Start timer, interval 1ms (will try to update as fast as possible)
        return True

    def update_frame(self):
        success, frame = self.cap.read()
        if success:
            # Calculate processing FPS
            curr_time = time.time()
            self.current_fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = curr_time
            # Log calculated FPS less frequently
            if 'frame_count' not in self.__dict__:
                self.frame_count = 0
            self.frame_count += 1
            if self.frame_count % 60 == 0:
                logging.debug(f"Calculated FPS in update_frame: {self.current_fps:.2f}")

            # Process frame using the tracker (now calls process_frame within Tracker)
            annotated_frame, results = self.tracker.process_frame(frame, self.current_fps, self.actual_fps_from_cap)

            # Update text labels with current data from the tracker
            self.fps_label.setText(f"FPS: {int(self.current_fps)} ({int(self.actual_fps_from_cap)} cap)")
            if results: # Ensure results are available before accessing tracker counts
                 self.scissors_label.setText(f"Scissors Reps: {self.tracker.scissors_reps}")
                 self.head_tilt_label.setText(f"Head Tilt Reps: {self.tracker.head_tilt_reps}")
                 self.head_rot_label.setText(f"Head Rot Reps: {self.tracker.head_rot_reps}")
                 self.bicep_left_label.setText(f"Bicep Left: {self.tracker.bicep_curl_reps_left}")
                 self.bicep_right_label.setText(f"Bicep Right: {self.tracker.bicep_curl_reps_right}")
                 self.overhead_press_label.setText(f"Overhead Press: {self.tracker.overhead_press_reps}")
                 self.triceps_ext_label.setText(f"Triceps Ext: {self.tracker.triceps_ext_reps}")
                 self.pushups_label.setText(f"Pushups: {self.tracker.pushup_reps}")
                 self.rows_label.setText(f"Rows: {self.tracker.row_reps}")

            # Convert the annotated frame to QImage
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_BGR888)

            # Display the QImage in the QLabel
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        # Release the camera when the window is closed
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.tracker.close()
        logging.info("Application finished and cleaned up.")
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self, tracker):
        super().__init__()

        self.setWindowTitle("Gym Tracking")
        self.setGeometry(100, 100, 800, 600) # Default window size

        self.video_widget = VideoStreamWidget(tracker)
        self.setCentralWidget(self.video_widget)

    def start_video(self, camera_index):
        if not self.video_widget.start_tracking(camera_index):
             # Handle camera opening failure, maybe show an error message
             logging.error("Failed to start video tracking.")
             self.close()


def main():
    logging.info("Application started.")

    app = QApplication(sys.argv)

    # Initialize pose tracker
    tracker = PoseTracker()

    # Camera selection (using the existing select_camera logic)
    camera_index = select_camera()
    if camera_index == -1:
        logging.error("Camera selection failed or no camera found. Exiting.")
        sys.exit(-1)

    main_window = MainWindow(tracker)
    main_window.show()

    # Start video capture after the window is shown
    main_window.start_video(camera_index)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 