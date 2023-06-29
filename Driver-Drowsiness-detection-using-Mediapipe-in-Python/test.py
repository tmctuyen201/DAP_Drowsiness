import cv2
import time
import numpy as np
import av
import numpy as np
from pydub import AudioSegment
import mediapipe as mp
import threading
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio

    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class VideoFrameHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
        }

        self.EAR_txt_pos = (10, 30)

    def process(self, video_capture, thresholds):
        """
        This function is used to implement our Drowsy detection algorithm

        Args:
            video_capture: (cv2.VideoCapture) VideoCapture object to capture frames
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH.

        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """

        ret, frame = video_capture.read()

        if not ret:
            return None, False

        frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])

            if EAR < thresholds["EAR_THRESH"]:

                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()

                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])
                    # playsound.playsound('/Users/daoxuanbac/Desktop/Python/SU23_DAP/Driver-Drowsiness-detection-using-Mediapipe-in-Python/audio/wake_up.wav')


            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])

        return frame, self.state_tracker["play_alarm"]

class AudioFrameHandler:
    """To play/pass custom audio based on some event"""

    def __init__(self, sound_file_path: str = ""):

        self.custom_audio = AudioSegment.from_file(file=sound_file_path, format="wav")
        self.custom_audio_len = len(self.custom_audio)

        self.ms_per_audio_segment: int = 20
        self.audio_segment_shape: tuple

        self.play_state_tracker: dict = {"curr_segment": -1}  # Currently playing segment
        self.audio_segments_created: bool = False
        self.audio_segments: list = []

    def prepare_audio(self, frame: av.AudioFrame):
        raw_samples = frame.to_ndarray()
        sound = AudioSegment(
            data=raw_samples.tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels),
        )

        self.ms_per_audio_segment = len(sound)
        self.audio_segment_shape = raw_samples.shape

        self.custom_audio = self.custom_audio.set_channels(sound.channels)
        self.custom_audio = self.custom_audio.set_frame_rate(sound.frame_rate)
        self.custom_audio = self.custom_audio.set_sample_width(sound.sample_width)

        self.audio_segments = [
            self.custom_audio[i : i + self.ms_per_audio_segment]
            for i in range(0, self.custom_audio_len - self.custom_audio_len % self.ms_per_audio_segment, self.ms_per_audio_segment)
        ]
        self.total_segments = len(self.audio_segments) - 1  # -1 because we start from 0.

        self.audio_segments_created = True

    def process(self, frame: av.AudioFrame, play_sound: bool = False):

        """
        Takes in the current input audio frame and based on play_sound boolean value
        either starts sending the custom audio frame or dampens the frame wave to emulate silence.

        For eg. playing a notification based on some event.
        """

        if not self.audio_segments_created:
            self.prepare_audio(frame)

        raw_samples = frame.to_ndarray()
        _curr_segment = self.play_state_tracker["curr_segment"]

        if play_sound:
            if _curr_segment < self.total_segments:
                _curr_segment += 1
            else:
                _curr_segment = 0

            sound = self.audio_segments[_curr_segment]

        else:
            if -1 < _curr_segment < self.total_segments:
                _curr_segment += 1
                sound = self.audio_segments[_curr_segment]
            else:
                _curr_segment = -1
                sound = AudioSegment(
                    data=raw_samples.tobytes(),
                    sample_width=frame.format.bytes,
                    frame_rate=frame.sample_rate,
                    channels=len(frame.layout.channels),
                )
                sound = sound.apply_gain(-100)

        self.play_state_tracker["curr_segment"] = _curr_segment

        channel_sounds = sound.split_to_mono()
        channel_samples = [s.get_array_of_samples() for s in channel_sounds]

        new_samples = np.array(channel_samples).T

        new_samples = new_samples.reshape(self.audio_segment_shape)
        new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate

        return new_frame


def main():
    # Set the thresholds for waiting time and EAR threshold.
    thresholds = {"WAIT_TIME": 2, "EAR_THRESH": 0.2}

    # Create an instance of VideoFrameHandler
    video_handler = VideoFrameHandler()

    # Create an instance of AudioFrameHandler
    audio_handler = AudioFrameHandler(sound_file_path='/Users/daoxuanbac/Desktop/Python/SU23_DAP/Driver-Drowsiness-detection-using-Mediapipe-in-Python/audio/wake_up.wav')

    # Create VideoCapture object for the webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Cannot open webcam")
        exit()

    # Infinite loop to process video frames
    while True:

        frame, play_alarm = video_handler.process(video_capture, thresholds)
        # If no frame is returned, exit the loop
        if frame is None:
            break

        cv2.imshow("Drowsiness Detection", frame)

        # Process audio frame
        audio_frame = av.AudioFrame.from_ndarray(np.zeros((1, 1)).astype('int16'), format="s16")

        audio_frame.sample_rate = 44100  # Replace with the appropriate sample rate

        audio_frame = audio_handler.process(audio_frame, play_sound=play_alarm)

        # Play alarm if flag is set
        if play_alarm:
            # Replace this with your alarm playing code
            print("Wake up! Wake up!")
            # Example: pygame.mixer.music.play()

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture and close windows
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


