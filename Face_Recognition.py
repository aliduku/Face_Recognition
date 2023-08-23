import face_recognition
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Load a sample picture and learn how to recognize it.
ali_image = face_recognition.load_image_file("aliEssam.jpg")
ali_face_encoding = face_recognition.face_encodings(ali_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    ali_face_encoding
]

known_face_names = [
    "Ali Essam"
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.processed_frame = None

    def recv(self, frame):
        # Convert to ndarray
        frame_data = frame.to_ndarray(format="bgr24")

        # Resize frame for faster face recognition processing
        small_frame = cv2.resize(frame_data, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        # Display the results on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame_data, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame_data, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_data, name, (left + 4, bottom - 4), font, 1, (0, 0, 255), 2)

        # Convert the modified frame to av.VideoFrame
        self.processed_frame = av.VideoFrame.from_ndarray(frame_data, format='bgr24')
        return self.processed_frame

def app():
    st.title("Face Recognition using Face_Recognition Library")
    st.write("Press the button below to start recognizing faces from your webcam")

    # Initialize the webrtc_streamer with the FaceRecognitionProcessor class
    webrtc_ctx = webrtc_streamer(
        key="face-recognition",
        video_processor_factory=lambda: FaceRecognitionProcessor(),
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        )

if __name__ == "__main__":
    app()
