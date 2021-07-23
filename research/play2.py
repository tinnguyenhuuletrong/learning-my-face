import cv2
import numpy as np
from play1 import face_detect, draw_face_rec, compare, load_model
import gradio as gr

id = "3"


def rec_and_draw_frame(frame, model):
    [face_locations, face_encodings] = face_detect(frame)
    if len(face_locations) > 0:
        i = 0
        for enc in face_encodings:
            res = compare([enc], "3", debug=False,
                          representations=model)
            color = (0, 255, 0) if res == True else (255, 0, 0)
            draw_face_rec(frame, [face_locations[i]], color)
            i += 1


def run_wc():
    model = load_model(id)
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        if ret == True:
            rec_and_draw_frame(frame, model)
            cv2.imshow("wc", frame)
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()


def run_gradio():
    model = load_model(id)

    def process(frame):
        rec_and_draw_frame(frame, model)
        return frame

    iface = gr.Interface(process, gr.inputs.Image(shape=None), "image")
    iface.launch()


# run_wc()
run_gradio()
