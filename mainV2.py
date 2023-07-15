import streamlit as st
import cv2
import torch
import numpy as np

def main():
    st.set_page_config(page_title="MIDAS Depth Estimation")
    st.title("MIDAS CV Depth Estimation")

    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    midas.to('cpu')
    midas.eval()

    transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transform.small_transform

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framebatch = transform(frame).to('cpu')
        with torch.no_grad():
            prediction = midas(framebatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = frame.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
            output = prediction.cpu().numpy()
            output = (output - np.min(output)) / (np.max(output) - np.min(output))
        frame_placeholder.image(output,channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
