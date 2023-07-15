# Necessary Imports
import cv2
import torch
import matplotlib.pyplot as plt

# Download MiDaS dataset
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Download transforms pipeline
transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transform.small_transform

# Hook into opencv
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make depth predictions
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

        print(output)

    plt.imshow(output)
    cv2.imshow('CV2Frame', frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows

plt.show()
