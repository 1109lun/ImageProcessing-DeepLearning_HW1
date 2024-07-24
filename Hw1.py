from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets
import sys
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchsummary
import torch.nn as nn
import torch.optim as optim

app = QtWidgets.QApplication(sys.argv) 
global pic1, blue_channel, red_channel, green_channel
filter_type = 0
window = QtWidgets.QMainWindow()
window.setWindowTitle("HW1")
window.setGeometry(300, 100, 1000, 800) 
loadimage1 = QtWidgets.QPushButton("Load Image 1", window)
loadimage1.move(80, 250)
loadimage2 = QtWidgets.QPushButton("Load Image 2", window)
loadimage2.move(80, 410)
# Create Q1
groupBox1= QtWidgets.QGroupBox("1.Image Processing", window)
groupBox1.setFixedSize(250 , 230)
groupBox1.move(230, 30)
pushButton1_1 = QtWidgets.QPushButton("1.1 Color Separation", window)
pushButton1_1.resize(180, 40)
pushButton1_1.move(270, 60)
pushButton1_2 = QtWidgets.QPushButton("1.2 Color Transformation", window)
pushButton1_2.resize(180, 40)
pushButton1_2.move(270, 125)
pushButton1_3 = QtWidgets.QPushButton("1.3 Color Extraction", window)
pushButton1_3.resize(180, 40)
pushButton1_3.move(270, 190)
#Create Q2
groupBox2= QtWidgets.QGroupBox("2.Image Smoothing", window)
groupBox2.setFixedSize(250 , 230)
groupBox2.move(230, 280)
pushButton2_1 = QtWidgets.QPushButton("2.1 Guassian blur", window)
pushButton2_1.resize(180, 40)
pushButton2_1.move(270, 310)
pushButton2_2 = QtWidgets.QPushButton("2.2 Bilateral filter", window)
pushButton2_2.resize(180, 40)
pushButton2_2.move(270, 375)
pushButton2_3 = QtWidgets.QPushButton("2.3 Median filter", window)
pushButton2_3.resize(180, 40)
pushButton2_3.move(270, 440)
#Create Q3
groupBox3= QtWidgets.QGroupBox("3.Edge Detection", window)
groupBox3.setFixedSize(250 , 230)
groupBox3.move(230, 530)
pushButton3_1 = QtWidgets.QPushButton("3.1 Sobel X", window)
pushButton3_1.resize(180, 40)
pushButton3_1.move(270, 550)
pushButton3_2 = QtWidgets.QPushButton("3.2 Sobel Y", window)
pushButton3_2.resize(180, 40)
pushButton3_2.move(270, 600)
pushButton3_3 = QtWidgets.QPushButton("3.3 Combination and Threshold", window)
pushButton3_3.resize(200, 40)
pushButton3_3.move(260, 650)
pushButton3_4 = QtWidgets.QPushButton("3.4 Gradient Angle", window)
pushButton3_4.resize(180, 40)
pushButton3_4.move(270, 700)
#create Q4
groupBox4 = QtWidgets.QGroupBox("4.Transforms", window)
groupBox4.setFixedSize(250 , 290)
groupBox4.move(600, 30)
text4_1 = QtWidgets.QLineEdit(window)
text4_1.resize(130, 30)
text4_1.move(670 , 70)
text4_1.setText("0")
text4_2 = QtWidgets.QLineEdit(window)
text4_2.resize(130, 30)
text4_2.move(670, 120)
text4_2.setText("0")
text4_3 = QtWidgets.QLineEdit(window)
text4_3.resize(130, 30)
text4_3.move(670, 170)
text4_3.setText("0")
text4_4 = QtWidgets.QLineEdit(window)
text4_4.resize(130, 30)
text4_4.move(670, 220)
text4_4.setText("0")
label4_l = QtWidgets.QLabel("Rotation:", window)
label4_l.resize(60, 15)
label4_l.move(610, 80)
label4_2 = QtWidgets.QLabel("Scaling:", window)
label4_2.resize(60, 15)
label4_2.move(610, 130)
label4_3 = QtWidgets.QLabel("Tx:", window)
label4_3.resize(60, 15)
label4_3.move(610, 180)
label4_4 = QtWidgets.QLabel("Ty:", window)
label4_4.resize(60, 15)
label4_4.move(610, 230)
label4_5 = QtWidgets.QLabel("deg", window)
label4_5.resize(60, 15)
label4_5.move(810, 80)
label4_6 = QtWidgets.QLabel("pixel", window)
label4_6.resize(60, 15)
label4_6.move(810, 180)
label4_7 = QtWidgets.QLabel("pixel", window)
label4_7.resize(60, 15)
label4_7.move(810, 230)
pushButton4 = QtWidgets.QPushButton("4.Transforms", window)
pushButton4.resize(180, 40)
pushButton4.move(640, 270)
#Create Q5
groupBox5= QtWidgets.QGroupBox("5.VGG19", window)
groupBox5.setFixedSize(250 , 420)
groupBox5.move(600, 340)
pushButton5 = QtWidgets.QPushButton("Load Image", window)
pushButton5.resize(180, 40)
pushButton5.move(630, 360)
pushButton5_1 = QtWidgets.QPushButton("5.1 Show Agumented Images", window)
pushButton5_1.resize(200, 40)
pushButton5_1.move(620, 410)
pushButton5_2 = QtWidgets.QPushButton("5.2 Show Model Structure", window)
pushButton5_2.resize(180, 40)
pushButton5_2.move(630, 460)
pushButton5_3 = QtWidgets.QPushButton("5.3 Show Acc and Loss", window)
pushButton5_3.resize(180, 40)
pushButton5_3.move(630, 510)
pushButton5_4 = QtWidgets.QPushButton("5.4 Inference", window)
pushButton5_4.resize(180, 40)
pushButton5_4.move(630, 560)
label5 = QtWidgets.QLabel("Predict:", window)
label5.resize(60, 15)
label5.move(630, 610)

def loadimage1_clicked():
    global pic1
    pic1 = open_image_using_dialog()
    cv2.imshow("Image", pic1)
    cv2.waitKey(0)
    cv2.destroyWindow("Image")
loadimage1.clicked.connect(loadimage1_clicked)

def open_image_using_dialog():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly

    image_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)", options=options)
    print(image_path)
    if image_path:
        image = cv2.imread(image_path)
    else:
        image = None
    return image

def pushButton1_1_clicked():
    global pic1, blue_channel, red_channel, green_channel
    blue, green, red = cv2.split(pic1)
    zero_array = np.zeros_like(blue)
    blue_channel = cv2.merge((blue, zero_array, zero_array))
    green_channel = cv2.merge((zero_array, green, zero_array))
    red_channel = cv2. merge((zero_array, zero_array, red))
    cv2.imshow("B channel", blue_channel)
    cv2.imshow("G channel", green_channel)
    cv2.imshow("R channel", red_channel)
    cv2.waitKey(0)
    cv2.destroyWindow("R channel")
    cv2.waitKey(0)
    cv2.destroyWindow("G channel")
    cv2.waitKey(0)
    cv2.destroyWindow("B channel")
pushButton1_1.clicked.connect(pushButton1_1_clicked)
    
def pushButton1_2_clicked():
    global pic1, blue_channel, red_channel, green_channel
    gray_image_q1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_image_q2 = (blue_channel + red_channel + green_channel)//3     
    cv2.imshow("1_2_Q1", gray_image_q1)
    cv2.imshow("1_2_Q2", gray_image_q2)
    cv2.waitKey(0)
    cv2.destroyWindow("1_2_Q1")
    cv2.waitKey(0)
    cv2.destroyWindow("1_2_Q2") 
pushButton1_2.clicked.connect(pushButton1_2_clicked)

def pushButton1_3_clicked():
    global pic1
    lower_yellow = np.array([20, 20, 25])
    upper_yellow = np.array([40, 255, 255])
    lower_green = np.array([40, 20, 25])
    upper_green = np.array([80, 255, 255])
    hsv_image = cv2.cvtColor(pic1, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    yellow_green_mask = cv2.bitwise_or(yellow_mask, green_mask)
    non_yellow_green_mask = cv2.bitwise_not(yellow_green_mask)
    result_image = cv2.bitwise_and(pic1, pic1 , mask=non_yellow_green_mask)
    cv2.imshow("Yellow-Green Mask", yellow_green_mask)
    cv2.imshow("Image with Yellow and Green Removed", result_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Yellow-Green Mask") 
    cv2.waitKey(0)
    cv2.destroyWindow("Image with Yellow and Green Removed") 
pushButton1_3.clicked.connect(pushButton1_3_clicked)

radius = 3
def kernelsize_change(value):
    global radius
    radius = 2 * value + 1
    blurred_image = cv2.GaussianBlur(pic1, (radius, radius), 0)
    cv2.imshow("smoothing answer",blurred_image )
def pushButton2_1_clicked():
    global pic1
    cv2.namedWindow("Gaussian Blur")
    cv2.resizeWindow("Gaussian Blur", 2000, 200)
    cv2.createTrackbar("m", "Gaussian Blur", 1, 5, kernelsize_change)
    while True:
        key = cv2.waitKey(10)
        if key == 27:  
            break
    cv2.destroyAllWindows()
pushButton2_1.clicked.connect(pushButton2_1_clicked)

def kernelint_change(value):
    global radius
    radius = 2 * value + 1
    blurred_image = cv2.bilateralFilter(pic1, radius,90 , 90)
    cv2.imshow("smoothing answer",blurred_image )
def pushButton2_2_clicked():
    global pic1
    cv2.namedWindow("bilateralFilter")
    cv2.resizeWindow("bilateralFilter", 2000, 200)
    cv2.createTrackbar("m", "bilateralFilter", 1, 5, kernelint_change)
    while True:
        key = cv2.waitKey(10)
        if key == 27:  
            break
    cv2.destroyAllWindows()
pushButton2_2.clicked.connect(pushButton2_2_clicked)

def mediankernelint_change(value):
    global radius
    radius = 2 * value + 1
    blurred_image = cv2.medianBlur(pic1, radius)
    cv2.imshow("smoothing answer",blurred_image )
def pushButton2_3_clicked():
    global pic1
    cv2.namedWindow("medianBlur")
    cv2.resizeWindow("medianBlur", 2000, 200)
    cv2.createTrackbar("m", "medianBlur", 1, 5, mediankernelint_change)
    while True:
        key = cv2.waitKey(10)
        if key == 27: 
            break
    cv2.destroyAllWindows()
pushButton2_3.clicked.connect(pushButton2_3_clicked)

def pushButton3_1_clicked():
    global pic1 , sobel_x_image , sobel_x_image_abs
    gray_pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_pic1 = cv2.GaussianBlur(gray_pic1, (3, 3), 0)
    sobel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    height, width = gray_pic1.shape
    sobel_x_image = np.zeros_like(gray_pic1, dtype=np.float32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sobel_x_result = np.sum(gray_pic1[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            sobel_x_image[i, j] = sobel_x_result
    sobel_x_image_abs = np.abs(sobel_x_image).astype(np.uint8)
    cv2.imshow("Sobel X Image", sobel_x_image_abs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
pushButton3_1.clicked.connect(pushButton3_1_clicked)

def pushButton3_2_clicked():
    global pic1 , sobel_y_image , sobel_y_image_abs
    gray_pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_pic1 = cv2.GaussianBlur(gray_pic1, (3, 3), 0)
    sobel_y = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    height, width = gray_pic1.shape
    sobel_y_image = np.zeros_like(gray_pic1, dtype=np.float32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            sobel_y_result = np.sum(gray_pic1[i - 1:i + 2, j - 1:j + 2] * sobel_y)
            sobel_y_image[i, j] = sobel_y_result
    sobel_y_image_abs = np.abs(sobel_y_image).astype(np.uint8)
    cv2.imshow("Sobel Y Image", sobel_y_image_abs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
pushButton3_2.clicked.connect(pushButton3_2_clicked)

def pushButton3_3_clicked():
    global pic1 , sobel_x_image_abs , sobel_y_image_abs , combined_image
    sobel_x_image_abs = sobel_x_image_abs.astype(np.float32)
    sobel_y_image_abs = sobel_y_image_abs.astype(np.float32)
    combined_image = np.sqrt(np.square(sobel_x_image_abs) + np.square(sobel_y_image_abs))
    combined_image = (combined_image * 255 / np.max(combined_image)).astype(np.uint8)
    threshold_value = 128
    threshold_result = np.where(combined_image > threshold_value, 255, 0).astype(np.uint8)
    cv2.imshow("Combined Image", combined_image)
    cv2.imshow("Thresholded Image", threshold_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
pushButton3_3.clicked.connect(pushButton3_3_clicked)

def pushButton3_4_clicked():
    global sobel_x_image , sobel_y_image , combined_image
    gradient_angle = np.arctan2(sobel_y_image, sobel_x_image)
    gradient_angle = (np.degrees(gradient_angle) + 360) % 360
    mask1 = ((gradient_angle >= 120) & (gradient_angle <= 180)).astype(np.uint8) * 255
    mask2 = ((gradient_angle >= 210) & (gradient_angle <= 330)).astype(np.uint8) * 255
    result1 = cv2.bitwise_and(combined_image, combined_image, mask=mask1)
    result2 = cv2.bitwise_and(combined_image, combined_image, mask=mask2)
    cv2.imshow("Angle Range 120-180", result1)
    cv2.imshow("Angle Range 210-330", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
pushButton3_4.clicked.connect(pushButton3_4_clicked)

def pushButton4_clicked():
    global pic1
    angle = float(text4_1.text())
    scale = float(text4_2.text())
    translate_x = int(text4_3.text())
    translate_y = int(text4_4.text())
    height, width = pic1.shape[:2]
    C = (240, 200)
    C_new = (C[0] + translate_x, C[1] + translate_y)
    M_rotate = cv2.getRotationMatrix2D(C, angle, scale)
    translation_matrix = np.float32([[1, 0, C_new[0] - C[0]], [0, 1, C_new[1] - C[1]]])
    result_image = cv2.warpAffine(pic1, M_rotate, (width, height))
    answer = cv2.warpAffine(result_image , translation_matrix , (width , height))
    cv2.imshow("Transformed Burger", answer)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
pushButton4.clicked.connect(pushButton4_clicked)

def pushButton5_1_clicked():
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
        ])
        folder_path = "./Dataset_OpenCvDl_Hw1/Dataset_OpenCvDl_Hw1/Q5_image/Q5_1"
        image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png'))] # 找到folder_path內部結尾有.jpg或.png的檔案

        plt.figure(figsize=(12, 8))
        plt.ion()
        plt.clf()

        for i, image_file in enumerate(image_files[:9]):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path) 
            augmented_image = transform(image) 

            plt.subplot(3, 3, i + 1)
            plt.title(image_file.replace(".png", ""))
            plt.imshow(augmented_image)

        plt.show()
        plt.ioff()
pushButton5_1.clicked.connect(pushButton5_1_clicked)

def pushButton5_2_clicked():
    model = torchvision.models.vgg19_bn(num_classes=10)
    torchsummary.summary(model, (3, 32, 32))
pushButton5_2.clicked.connect(pushButton5_2_clicked)    

def pushButton5_3_clicked():
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define the VGG19 with batch normalization model, loss function, and optimizer
    model = torchvision.models.vgg19_bn(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Training and validation loop
    num_epochs = 40
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
    # Training loop
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

    for data in trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_loss_list.append(train_loss / len(trainloader))
    train_acc_list.append(train_accuracy)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    val_loss_list.append(val_loss / len(testloader))
    val_acc_list.append(val_accuracy)

    # Save the model with the highest validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

# Create a line chart for the training and validation loss and accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_list, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

# Save the figure as "loss_accuracy.png"
    plt.savefig('loss_accuracy.png')

# Display the saved figure
    plt.show()
pushButton5_3.clicked.connect(pushButton5_3_clicked) #can't run

window.show()

sys.exit(app.exec_())


    