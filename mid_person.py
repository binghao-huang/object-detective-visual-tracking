#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import threading
import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int64
import time
import os
import sys
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

yolo = YOLO()
bridge = CvBridge()
def send_right():
    rospy.Subscriber('/mid_camera/color/image_raw/compressed', CompressedImage, ReceiveVideo_right)
    rospy.spin()

def ReceiveVideo_right(data):
    global cv_image
    # print(1)
    cv_image = bridge.compressed_imgmsg_to_cv2(data, 'bgr8')
    # print(cv_image.shape)
    # cv2.imshow('right', cv_image)
    # k = cv2.waitKey(10)
    # if k ==27:     # 键盘上Esc键的键值
    #     cv2.destroyAllWindows()

    # cv2.imshow('image', cv_image)
def main():
    global cv_image, command_vel_pub_m
    time.sleep(4)
    fps = 0
    while(True):
        t1 = time.time()
        # 读取某一帧
        frame = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame, bbox_list, label_list = yolo.detect_image(frame)
        frame = np.array(frame)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(frame.shape)

        cv2.imshow("video",frame)

        c= cv2.waitKey(1) & 0xff 
        if c==27:
            break
        
        if type(label_list) != int: # 没检测到物体的时候，bbox_list和label_list为1
            num_of_obj = len(label_list)
            #print('num_of_object:', num_of_obj)
            #确定跟踪物体与图像中点的相对坐标
            for i in range(num_of_obj):
                if 'person' in label_list[i]:
                    object_center = (bbox_list[i][1]+bbox_list[i][3])*0.5
                    print('object_center:   ',object_center)
                    if(320-50<object_center<320+50):
                        print('midmidmijdmimdasdadasdasdasimafafafegfagsergerdi')
                        command_STOP_pub_m.publish(1)
                    #motor1_move()
                # elif 'dog' or 'cat' in label_list[i]:
                #     print("wangwang miaomiao")
                # pass
        else:
        	print('yolo未识别到任何物体')
        pass

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0] #获取文件名
    rospy.init_node(node_name) 

    global command_vel_pub_m
    #创建发布者
    command_STOP_pub_m = rospy.Publisher('/STOP_motor1_control', Int64, queue_size = 100, latch=True)
    
    t_send_right = threading.Thread(target = send_right)
    t_send_right.start()
    t_main = threading.Thread(target=main)
    t_main.start()