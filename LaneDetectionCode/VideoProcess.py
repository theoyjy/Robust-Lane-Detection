import cv2
from model import generate_model
from config import args_setting
import config
import numpy as np
import torch
from PIL import Image
import time


video_path = "path/to/video.mp4"

if __name__ == '__main__':
    
    frames = []

    cap = cv2.VideoCapture(video_path)
    args = args_setting()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = generate_model(args)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    frame_duration = 1 / fps
    delay = int(frame_duration * 1000)  # Delay in milliseconds


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Preprocess the frame
        frame = cv2.resize(frame, (128, 256))

        frames.append(frame)
        if len(frames) < 5:
            continue

        if len(frames) > 5:
            frames.pop(0)

        # Pass the frame to the model
        output = model(data)
            # feature_dic.append(feature)
        pred = output.max(1, keepdim=True)[1]
        img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
        img = Image.fromarray(img.astype(np.uint8))

        data = torch.squeeze(data).cpu().numpy()
        if args.model == 'SegNet-ConvLSTM' or args.model == 'UNet-ConvLSTM':
            data = np.transpose(data[-1], [1, 2, 0]) * 255
        else:
            data = np.transpose(data, [1, 2, 0]) * 255
        data = Image.fromarray(data.astype(np.uint8))
        rows = img.size[0]
        cols = img.size[1]
        for i in range(0, rows):
            for j in range(0, cols):
                img2 = (img.getpixel((i, j)))
                if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                    data.putpixel((i, j), (234, 53, 57, 255))
        data = data.convert("RGB")
        # data.save(config.save_path + "%s_data.jpg" % k)#red line on the original image
        # img.save(config.save_path + "%s_pred.jpg" % k)#prediction result

        processing_time = time.time() - start_time
        adjusted_delay = max(1, int(delay - processing_time * 1000)) 
        # Display the result
        cv2.imshow("Lane Detection", data)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()