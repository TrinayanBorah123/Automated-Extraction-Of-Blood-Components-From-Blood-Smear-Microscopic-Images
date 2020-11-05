from Tkinter import *
import tkFileDialog
import Tkinter as tk
from PIL import Image,ImageTk,ImageEnhance 
import cv2
import numpy as np
from skimage import filters
import skfuzzy as fuzz
import re
root=Tk()
root.title("Extraction of WBC and Platelet")
root.geometry("1600x800")
#root.filename="D:/My Study/7th Semester/Major Project/square.jpg"
#path="D:/My Study/7th Semester/Major Project/square.jpg"
#print(root.filename)
sele=False
chan=False
sege=False
newWindow=""
ch_type=""
gd_path=""
tp=0
tn=0
fp=0
fn=0
accuracy=0
specificity=0
precision=0
recall=0
s=0
p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
def createNewWindow(gd):
    newWindow = tk.Toplevel(root)
    newWindow.geometry("1420x640")
    #print gd
    gd_pl=gd[-7:]
    for s in re.findall(r'-?\d+\.?\d*', gd_pl):
       print( s) 
    location0="D:/My Study/7th Semester/Major Project/Work/k_bit_ero_OUTPUT_ch1.jpg"
    sublbl0=Label(newWindow,text="ORIGINAL KMEANS OUTPUT")
    sublbl0.place(x=40,y=20)
    load = Image.open(location0)
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img0 = Label(newWindow, image=render,width="250",height="250")
    img0.image = render
    img0.place(x=40, y=50)
    k_count=0
    k_p_count=0
    k_wbc_count=0
    location1="D:/My Study/7th Semester/Major Project/Work/f_bit_ero_OUTPUT_ch1.jpg"
    sublbl1=Label(newWindow,text="ORIGINAL FUZZY C MEAN OUTPUT")
    sublbl1.place(x=40,y=320)
    load = Image.open(location1)
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img1 = Label(newWindow, image=render,width="250",height="250")
    img1.image = render
    img1.place(x=40, y=340)
    #WBC and Platelet separation fro K means
    im = Image.open(location0)
    im.save('D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1.png','PNG')
    src = cv2.imread('D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1.png', cv2.IMREAD_GRAYSCALE)
    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(src,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S) 
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 4000:   #keep
            result[labels == i + 1] = 255
            k_wbc_count+=1
    #print "WBC:",wbc_count
    cv2.imwrite("D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1_wbc.png", result)
    img1 = cv2.imread('D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1_wbc.png', cv2.IMREAD_GRAYSCALE)
    dest_xor = cv2.bitwise_xor(img1, img2, mask = None)
    cv2.imwrite("D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1_platelet.png", dest_xor)
    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(dest_xor,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S) 
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 30:   #keep
            result[labels == i + 1] = 255
            k_count+=1
    k_p_count=k_count-k_wbc_count
    print ("total:",k_count)
    print ("wbc:",k_wbc_count)
    print ("Platelet:",k_p_count)
    sublbl0=Label(newWindow,text="FOR K-MEANS: ")
    sublbl0.place(x=960,y=20)
    sublbl0=Label(newWindow,text="NUMBER OF WBC: ")
    sublbl0.place(x=960,y=50)
    sublbl1=Label(newWindow,text=k_wbc_count)
    sublbl1.place(x=1100,y=50)
    
    sublbl0=Label(newWindow,text="NUMBER OF PLATELETS: ")
    sublbl0.place(x=960,y=70)
    sublbl1=Label(newWindow,text=k_p_count)
    sublbl1.place(x=1100,y=70)
    
    sublbl0=Label(newWindow,text="WBC OF KMEANS")
    sublbl0.place(x=340,y=20)
    load = Image.open("D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1_wbc.png")
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img1 = Label(newWindow, image=render,width="250",height="250")
    img1.image = render
    img1.place(x=360, y=50)
    sublbl0=Label(newWindow,text="PLATELETS OF KMEANS")
    sublbl0.place(x=660,y=20)
    load = Image.open("D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1_platelet.png")
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img1 = Label(newWindow, image=render,width="250",height="250")
    img1.image = render
    img1.place(x=660, y=50)
 
    #WBC and Platelet separation for Fuzzy
    f_wbc_count=0
    f_p_count=0
    f_count=0
    im = Image.open(location1)
    im.save('D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1.png','PNG')
    src = cv2.imread('D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1.png', cv2.IMREAD_GRAYSCALE)

    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(src,127,255,0)
    
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 4000:   #keep
            result[labels == i + 1] = 255
            f_wbc_count+=1
    
    cv2.imwrite("D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1_wbc.png", result)
    img1 = cv2.imread('D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1_wbc.png', cv2.IMREAD_GRAYSCALE)
    dest_xor = cv2.bitwise_xor(img1, img2, mask = None)
    cv2.imwrite("D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1_platelet.png", dest_xor)
    ret, binary_map = cv2.threshold(dest_xor,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S) 
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 30:   #keep
            result[labels == i + 1] = 255
            f_count+=1
    f_p_count=f_count-f_wbc_count
    
    sublbl1=Label(newWindow,text="WBC OF FUZZY C MEAN")
    sublbl1.place(x=340,y=320)
    load = Image.open("D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1_wbc.png")
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img1 = Label(newWindow, image=render,width="250",height="250")
    img1.image = render
    img1.place(x=360, y=340)
    sublbl1=Label(newWindow,text="PLATELETS OF FUZZY C MEAN")
    sublbl1.place(x=660,y=320)
    load = Image.open("D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1_platelet.png")
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img1 = Label(newWindow, image=render,width="250",height="250")
    img1.image = render
    img1.place(x=660, y=340)
    
    sublbl0=Label(newWindow,text="FOR FUZZY C MEANS: ")
    sublbl0.place(x=960,y=320)
    sublbl0=Label(newWindow,text="NUMBER OF WBC: ")
    sublbl0.place(x=960,y=350)
    sublbl1=Label(newWindow,text=f_wbc_count)
    sublbl1.place(x=1100,y=350)
    
    sublbl0=Label(newWindow,text="NUMBER OF PLATELETS: ")
    sublbl0.place(x=960,y=370)
    sublbl1=Label(newWindow,text=f_p_count)
    sublbl1.place(x=1100,y=370)
    
    
    ##WBC and Platelets count Ground Truth
    gd_platelet_count=0
    gd_wbc_count=0
    #gd_platelet_load = Image.open("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_platelet/ground"+gd_pl)
    
    gd_platelet_load = cv2.imread("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_platelet/ground"+s+"png",cv2.IMREAD_GRAYSCALE)
    #gd_wbc_load = Image.open("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_wbc/ground"+gd_pl)
    ret, binary_map = cv2.threshold(gd_platelet_load,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S) 
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 100:   #keep
            result[labels == i + 1] = 255
            gd_platelet_count+=1
    #print gd_platelet_count
    gd_wbc_load = cv2.imread("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_wbc/ground"+s+"png",cv2.IMREAD_GRAYSCALE)
    # convert to binary by thresholding
    ret, binary_map = cv2.threshold(gd_wbc_load,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 4000:   #keep
            result[labels == i + 1] = 255
            gd_wbc_count+=1
    #print gd_wbc_count
    sublbl0=Label(newWindow,text="NUMBER OF WBC IN GROUNDTRUTH: ")
    sublbl0.place(x=960,y=90)
    sublbl1=Label(newWindow,text=gd_wbc_count)
    sublbl1.place(x=1200,y=90)
    sublbl0=Label(newWindow,text="NUMBER OF PLATELETS IN GROUNDTRUTH: ")
    sublbl0.place(x=960,y=110)
    sublbl1=Label(newWindow,text=gd_platelet_count)
    sublbl1.place(x=1200,y=110)
    
    sublbl0=Label(newWindow,text="NUMBER OF WBC IN GROUNDTRUTH: ")
    sublbl0.place(x=960,y=390)
    sublbl1=Label(newWindow,text=gd_wbc_count)
    sublbl1.place(x=1200,y=390)
    sublbl0=Label(newWindow,text="NUMBER OF PLATELETS IN GROUNDTRUTH: ")
    sublbl0.place(x=960,y=410)
    sublbl1=Label(newWindow,text=gd_platelet_count)
    sublbl1.place(x=1200,y=410)
    
    #Calculation of different factores for Platelets K-Means
    #performing AND operation between above images...
    k_count_of_and_oper=0
   
    #specificity=0
    #precision=0
    #recall=0
    #accuracy=0
    k_ground_img = cv2.imread("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_platelet/ground"+s+"png",cv2.IMREAD_GRAYSCALE)
    k_segmented_img = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1_platelet.png", cv2.IMREAD_GRAYSCALE)
    k_ground_img_and_segmented_img = cv2.bitwise_and(k_ground_img, k_segmented_img, mask = None)
    #plt.imshow(dest_and)
    cv2.imwrite(r"D:/My Study/7th Semester/Major Project/Work/k_ground_img_and_segmented_img.png", k_ground_img_and_segmented_img)
    
    #counting the platelet in 'and'coperated image...
    k_src = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/k_ground_img_and_segmented_img.png", cv2.IMREAD_GRAYSCALE)
    ret, binary_map = cv2.threshold(k_src,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >=100:   #keep
            result[labels == i + 1] = 255
            k_count_of_and_oper=k_count_of_and_oper+1
        
    print("K-MEANS AND OPERATION COUNT=",k_count_of_and_oper)
        
    tp=k_count_of_and_oper
    fp=k_p_count-k_count_of_and_oper
    if(fp<0): 
        fp=0
    fn=gd_platelet_count-k_count_of_and_oper
    if(fn<0):
        fn=0
    tn=gd_platelet_count-(tp+fp+fn);
    if(tn<0):
        tn=1
    #print("TP=",tp)
    #print("FP=",fp)
    #print("FN=",fn)
    #print("TN=",tn)
    tn=1
    #Accuracy calculation
    #Accuracy calculation
    accuracy=float(float(tp+tn)/float(tp+tn+fp+fn))*100
    #accuracy_sum=accuracy_sum+accuracy
    #Specificity calculation
    specificity=float(float(tn)/float(tn+fp))*100
    #specificity_sum=specificity_sum+specificity
    #Precision calculation
    if(tp==0):
        precision=0
    else:
        precision=float(float(tp)/float(tp+fp))*100
    #precision_sum=precision+precision_sum
    #Recall calculation
    recall=float(float(tp)/float(tp+fn))*100
    #recall_sum=recall_sum+recall
    #print("Accuracy=",accuracy)
    #print("Specificity=",specificity)
    #print("Precision=",precision)
    #print("Recall=",recall)
    
    #sublbl0=Label(newWindow,text="ACCURACY     SPECIFICITY     PRECISION     RECALL")
    #sublbl0.place(x=1020,y=130)
    sublbl1=Label(newWindow,text="PLATELET:  "+"{:.2f}".format(accuracy)+"(ACCURACY), "+"{:.2f}".format(specificity)+"(SPECIFICITY), "+"{:.2f}".format(precision)+"(PRECISION), "+"{:.2f}".format(recall)+"(RECALL)")
    sublbl1.place(x=960,y=130)
    
    #Calculation of different factores for Platelets Fuzzy
    #performing AND operation between above images...
    f_count_of_and_oper=0
    f_ground_img = cv2.imread("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_platelet/ground"+s+"png",cv2.IMREAD_GRAYSCALE)
    f_segmented_img = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1_platelet.png", cv2.IMREAD_GRAYSCALE)
    f_ground_img_and_segmented_img = cv2.bitwise_and(f_ground_img, f_segmented_img, mask = None)
    #plt.imshow(dest_and)
    cv2.imwrite(r"D:/My Study/7th Semester/Major Project/Work/f_ground_img_and_segmented_img.png", f_ground_img_and_segmented_img)
    
    #counting the platelet in 'and'coperated image...
    f_src = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/f_ground_img_and_segmented_img.png", cv2.IMREAD_GRAYSCALE)
    ret, binary_map = cv2.threshold(f_src,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >=100:   #keep
            result[labels == i + 1] = 255
            f_count_of_and_oper=f_count_of_and_oper+1
        
    print("FUZZY AND OPERATION COUNT=",f_count_of_and_oper)
        
    tp=f_count_of_and_oper
    fp=f_p_count-f_count_of_and_oper
    if(fp<0):
        fp=0
    fn=gd_platelet_count-f_count_of_and_oper
    if(fn<0):
        fn=0
    tn=gd_platelet_count-(tp+fp+fn);
    if(tn<0):
        tn=1
    #print("TP=",tp)
    #print("FP=",fp)
    #print("FN=",fn)
    #print("TN=",tn)
    tn=1
    #Accuracy calculation
    accuracy=float(float(tp+tn)/float(tp+tn+fp+fn))*100
    #accuracy_sum=accuracy_sum+accuracy
    #Specificity calculation
    specificity=float(float(tn)/float(tn+fp))*100
    #specificity_sum=specificity_sum+specificity
    #Precision calculation
    if(tp==0):
        precision=0
    else:
        precision=float(float(tp)/float(tp+fp))*100
    #precision_sum=precision+precision_sum
    #Recall calculation
    recall=float(float(tp)/float(tp+fn))*100
    #recall_sum=recall_sum+recall
    #print("Accuracy="+"{:.2f}".format(accuracy))
    #print("Specificity="+"{:.2f}".format(specificity))
    #print("Precision="+"{:.2f}".format(precision))
    #print("Recall="+"{:.2f}".format(recall))
    #sublbl0=Label(newWindow,text="ACCURACY     SPECIFICITY     PRECISION     RECALL")
    #sublbl0.place(x=1020,y=430)
    sublbl1=Label(newWindow,text="PLATELET:  "+"{:.2f}".format(accuracy)+"(ACCURACY), "+"{:.2f}".format(specificity)+"(SPECIFICITY), "+"{:.2f}".format(precision)+"(PRECISION), "+"{:.2f}".format(recall)+"(RECALL)")
    sublbl1.place(x=960,y=430)
    
    #Calculation of different factores for WBC K-Means
    #performing AND operation between above images...
    k_count_of_and_oper=0
   
    #specificity=0
    #precision=0
    #recall=0
    #accuracy=0
    k_ground_img = cv2.imread("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_wbc/ground"+s+"png",cv2.IMREAD_GRAYSCALE)
    k_segmented_img = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/k_newpng_ero_OUTPUT_ch1_wbc.png", cv2.IMREAD_GRAYSCALE)
    k_ground_img_and_segmented_img = cv2.bitwise_and(k_ground_img, k_segmented_img, mask = None)
    #plt.imshow(dest_and)
    cv2.imwrite(r"D:/My Study/7th Semester/Major Project/Work/k_ground_img_and_segmented_img_WBC.png", k_ground_img_and_segmented_img)
    
    #counting the platelet in 'and'coperated image...
    k_src = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/k_ground_img_and_segmented_img_WBC.png", cv2.IMREAD_GRAYSCALE)
    ret, binary_map = cv2.threshold(k_src,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >=100:   #keep
            result[labels == i + 1] = 255
            k_count_of_and_oper=k_count_of_and_oper+1
        
    print("K-MEANS AND OPERATION COUNT=",k_count_of_and_oper)
        
    tp=k_count_of_and_oper
    fp=k_wbc_count-k_count_of_and_oper
    if(fp<0): 
        fp=0
    fn=gd_wbc_count-k_count_of_and_oper
    if(fn<0):
        fn=0
    tn=gd_wbc_count-(tp+fp+fn);
    if(tn<0):
        tn=1
    #print("TP=",tp)
    #print("FP=",fp)
    #print("FN=",fn)
    #print("TN=",tn)
    tn=1
    #Accuracy calculation
    #Accuracy calculation
    accuracy=float(float(tp+tn)/float(tp+tn+fp+fn))*100
    #accuracy_sum=accuracy_sum+accuracy
    #Specificity calculation
    specificity=float(float(tn)/float(tn+fp))*100
    #specificity_sum=specificity_sum+specificity
    #Precision calculation
    if(tp==0):
        precision=0
    else:
        precision=float(float(tp)/float(tp+fp))*100
    #precision_sum=precision+precision_sum
    #Recall calculation
    recall=float(float(tp)/float(tp+fn))*100
    #recall_sum=recall_sum+recall
    #print("Accuracy=",accuracy)
    #print("Specificity=",specificity)
    #print("Precision=",precision)
    #print("Recall=",recall)
    
    #sublbl0=Label(newWindow,text="ACCURACY     SPECIFICITY     PRECISION     RECALL")
    #sublbl0.place(x=1020,y=130)
    sublbl1=Label(newWindow,text="WBC:   "+"{:.2f}".format(accuracy)+"(ACCURACY), "+"{:.2f}".format(specificity)+"(SPECIFICITY), "+"{:.2f}".format(precision)+"(PRECISION), "+"{:.2f}".format(recall)+"(RECALL)")
    sublbl1.place(x=960,y=150)
    
    #Calculation of different factores for WBC FUZZY
    #performing AND operation between above images...
    f_count_of_and_oper=0
   
    #specificity=0
    #precision=0
    #recall=0
    #accuracy=0
    f_ground_img = cv2.imread("D:/My Study/8th Semester/Major Project/ground_truth/ground_of_wbc/ground"+s+"png",cv2.IMREAD_GRAYSCALE)
    f_segmented_img = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/f_newpng_ero_OUTPUT_ch1_wbc.png", cv2.IMREAD_GRAYSCALE)
    f_ground_img_and_segmented_img = cv2.bitwise_and(f_ground_img, f_segmented_img, mask = None)
    #plt.imshow(dest_and)
    cv2.imwrite(r"D:/My Study/7th Semester/Major Project/Work/f_ground_img_and_segmented_img_WBC.png", f_ground_img_and_segmented_img)
    
    #counting the platelet in 'and'coperated image...
    f_src = cv2.imread(r"D:/My Study/7th Semester/Major Project/Work/f_ground_img_and_segmented_img_WBC.png", cv2.IMREAD_GRAYSCALE)
    ret, binary_map = cv2.threshold(f_src,127,255,0)
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >=100:   #keep
            result[labels == i + 1] = 255
            f_count_of_and_oper=f_count_of_and_oper+1
        
    print("FUZZY AND OPERATION COUNT=",f_count_of_and_oper)
        
    tp=f_count_of_and_oper
    fp=f_wbc_count-f_count_of_and_oper
    if(fp<0): 
        fp=0
    fn=gd_wbc_count-f_count_of_and_oper
    if(fn<0):
        fn=0
    tn=gd_wbc_count-(tp+fp+fn);
    if(tn<0):
        tn=1
    print("TP=",tp)
    print("FP=",fp)
    print("FN=",fn)
    print("TN=",tn)
    tn=1
    #Accuracy calculation
    #Accuracy calculation
    accuracy=float(float(tp+tn)/float(tp+tn+fp+fn))*100
    #accuracy_sum=accuracy_sum+accuracy
    #Specificity calculation
    specificity=float(float(tn)/float(tn+fp))*100
    #specificity_sum=specificity_sum+specificity
    #Precision calculation
    if(tp==0):
        precision=0
    else:
        precision=float(float(tp)/float(tp+fp))*100
    #precision_sum=precision+precision_sum
    #Recall calculation
    recall=float(float(tp)/float(tp+fn))*100
    #recall_sum=recall_sum+recall
    print("Accuracy=",accuracy)
    print("Specificity=",specificity)
    print("Precision=",precision)
    print("Recall=",recall)
    
    #sublbl0=Label(newWindow,text="ACCURACY     SPECIFICITY     PRECISION     RECALL")
    #sublbl0.place(x=1020,y=130)
    sublbl1=Label(newWindow,text="WBC:   "+"{:.2f}".format(accuracy)+"(ACCURACY), "+"{:.2f}".format(specificity)+"(SPECIFICITY), "+"{:.2f}".format(precision)+"(PRECISION), "+"{:.2f}".format(recall)+"(RECALL)")
    sublbl1.place(x=960,y=450)
    
    
def imshow(location,x1,y1):
    load = Image.open(location)
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = Label(root, image=render,width="250",height="250")
    img.image = render
    img.place(x=x1, y=y1)

    
def kmeans(cpath,valtype):
    cimg=Image.open(cpath)
    br_en=ImageEnhance.Brightness(cimg)
    br_image=br_en.enhance(2.0)
    br_image.save('D:/My Study/7th Semester/Major Project/Work/br_enhance.jpg','JPEG')
    br_cimg=cv2.imread(r"D:\My Study\7th Semester\Major Project\Work\br_enhance.jpg")
    image = cv2.cvtColor(br_cimg, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 3
    compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # disable only the cluster number 2 (turn the pixel into black)
    
    max_val=255
    ret,im1=cv2.threshold(segmented_image,145,max_val,cv2.THRESH_BINARY)
    kernal=np.ones((4,4),np.uint8)
    kernal1=np.ones((6,6),np.uint8)
    dilation=cv2.dilate(im1,kernal,iterations=2)
    erosion=cv2.erode(dilation,kernal1,iterations=2)
    cv2.imwrite(r"D:/My Study/7th Semester/Major Project/Work/k_bit_ero_"+valtype+"_ch1.jpg",erosion)
    location="D:/My Study/7th Semester/Major Project/Work/k_bit_ero_"+valtype+"_ch1.jpg"
    imshow(location,1060,155)
#Fuzzy C Mean
def fuzzy_c_mean(cpath,valtype):
    cimg=Image.open(cpath)
    br_en=ImageEnhance.Brightness(cimg)
    br_image=br_en.enhance(2.0)
    br_image.save('D:/My Study/7th Semester/Major Project/Work/f_br_enhance.jpg','JPEG')
    br_cimg=cv2.imread(r"D:\My Study\7th Semester\Major Project\Work\f_br_enhance.jpg")
    clusters=2
    image = cv2.cvtColor(br_cimg, cv2.COLOR_BGR2RGB)
    max_val=255
    th =filters.threshold_otsu(image)+78
    ret,img1=cv2.threshold(image,th,max_val,cv2.THRESH_BINARY)
    labels, centers = fcm_images(img1, clusters)
    creat_image(labels, centers,valtype)
def fcm_images(img, c):
    # Reshape the image into a 2D array
    data = img.reshape(img.shape[0]*img.shape[1], -1).T
    if data.shape[0] > 3:
        data = data[:3, :]
    # Fuzzy C-Means Clustering function call
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c, m=2, error=0.005, maxiter=1000, init=None)
    # Assign the maximum values of membership of each pixel to the 2D array
    labels = np.argmax(u, axis=0).reshape(img.shape[0], img.shape[1])
    # Create an image for each cluster
    return labels, cntr
def creat_image(labels, centers,valtype):
    img = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = centers[labels[i, j]]
    if(img.max() < 1):  #Made change >
        img /= 255
    #Errosion/Dilation
    kernal=np.ones((4,4),np.uint8)
    kernal1=np.ones((6,6),np.uint8)

    dilation=cv2.dilate(img,kernal,iterations=2)
    erosion=cv2.erode(dilation,kernal1,iterations=2)
    cv2.imwrite(r"D:/My Study/7th Semester/Major Project/Work/f_bit_ero_"+valtype+"_ch1.jpg",erosion)
    location="D:/My Study/7th Semester/Major Project/Work/f_bit_ero_"+valtype+"_ch1.jpg"
    imshow(location,1060,455)
    return img

def channelback(location,x1,y1):
    load = Image.open(location)
    load = load.resize((200, 200), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = Label(root, image=render,width="200",height="200")
    img.image = render
    img.place(x=x1, y=y1)
    
def click():
    #global flag; 
    #print flag
    
    global sele
    global chan
    global sege
    global gd_path
    
    root.filename=tkFileDialog.askopenfilename(initialdir="/",title="Select an Image",filetype=(("jpeg","jpg"),("all files","*.*")))
    gd_path=root.filename
    imshow(root.filename,40,130)
    #print root.filename
    #gd_path="HOOW"
    #print gd_path
    sele=True
    chan=False
    sege=False
    
    
    #flag=1
def channel():
    #print flag
    global sele
    global chan
    global sege
    if root.filename!=path and sele ==True:
        img1=cv2.imread(root.filename)
        red_channel=img1[:,:,0]
        green_channel=img1[:,:,1]
        blue_channel=img1[:,:,2]
        c=1-red_channel;
        m=1-green_channel;
        y=1-blue_channel;
        hsvimg=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
        h=hsvimg[:,:,0];
        s=hsvimg[:,:,1];
        v=hsvimg[:,:,2];
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\red_ch1.jpg",red_channel)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\green_ch1.jpg",green_channel)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\blue_ch1.jpg",blue_channel)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\cyan_ch1.jpg",c)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\magenta_ch1.jpg",m)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\yellow_ch1.jpg",y)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\hue_ch1.jpg",h)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\satu_ch1.jpg",s)
        cv2.imwrite(r"D:\My Study\7th Semester\Major Project\Work\value_ch1.jpg",v)
        location1="D:/My Study/7th Semester/Major Project/Work/red_ch1.jpg"
        location2="D:/My Study/7th Semester/Major Project/Work/green_ch1.jpg"
        location3="D:/My Study/7th Semester/Major Project/Work/blue_ch1.jpg"
        location4="D:/My Study/7th Semester/Major Project/Work/cyan_ch1.jpg"
        location5="D:/My Study/7th Semester/Major Project/Work/magenta_ch1.jpg"
        location6="D:/My Study/7th Semester/Major Project/Work/yellow_ch1.jpg"
        location7="D:/My Study/7th Semester/Major Project/Work/hue_ch1.jpg"
        location8="D:/My Study/7th Semester/Major Project/Work/satu_ch1.jpg"
        location9="D:/My Study/7th Semester/Major Project/Work/value_ch1.jpg"
        channelback(location1,340,140)
        channelback(location2,560,140)
        channelback(location3,780,140)
        channelback(location4,340,370)
        channelback(location5,560,370)
        channelback(location6,780,370)
        channelback(location7,340,600)
        channelback(location8,560,600)
        channelback(location9,780,600)
        sele=True
        chan=True
        sege=False
        
def stype():
    global sele
    global chan
    global sege
    if root.filename!=path and sele==True and chan==True:
        k1=variable.get()
        if variable.get() == "RED":
            #print(variable.get()+"1")
            path1="D:/My Study/7th Semester/Major Project/Work/red_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "GREEN": 
            path1="D:/My Study/7th Semester/Major Project/Work/green_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "BLUE": 
            path1="D:/My Study/7th Semester/Major Project/Work/blue_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "CYAN": 
            path1="D:/My Study/7th Semester/Major Project/Work/cyan_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "MAGENTA": 
            path1="D:/My Study/7th Semester/Major Project/Work/magenta_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "YELLOW": 
            path1="D:/My Study/7th Semester/Major Project/Work/yellow_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "HUE": 
            path1="D:/My Study/7th Semester/Major Project/Work/hue_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "SATURATION": 
            path1="D:/My Study/7th Semester/Major Project/Work/satu_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        elif variable.get() == "VALUE": 
            path1="D:/My Study/7th Semester/Major Project/Work/value_ch1.jpg"
            kmeans(path1,"OUTPUT")
            fuzzy_c_mean(path1,"OUTPUT")
        sele=True
        chan=True
        sege=True
def reset():
    global sele
    global chan
    global sege
    path="D:/My Study/7th Semester/Major Project/square.jpg"
    imshow(path,40,130)
    lblred=Label(root,text="RED")
    lblred.place(x=340,y=120)
    channelback(path,340,140)
    lblgr=Label(root,text="GREEN")
    lblgr.place(x=560,y=120)
    channelback(path,560,140)
    lblbl=Label(root,text="BLUE")
    lblbl.place(x=780,y=120)
    channelback(path,780,140)
    lblcy=Label(root,text="CYAN")
    lblcy.place(x=340,y=350)
    channelback(path,340,370)
    lblmg=Label(root,text="MAGENTA")
    lblmg.place(x=560,y=350)
    channelback(path,560,370)
    lblyl=Label(root,text="YELLOW")
    lblyl.place(x=780,y=350)
    channelback(path,780,370)
    lblhue=Label(root,text="HUE")
    lblhue.place(x=340,y=580)
    channelback(path,340,600)
    lblval=Label(root,text="SATURATION")
    lblval.place(x=560,y=580)
    channelback(path,560,600)
    lblsatu=Label(root,text="VALUE")
    lblsatu.place(x=780,y=580)
    channelback(path,780,600)         
    imshow(path,1060,155)
    channelback(path,780,600)         
    imshow(path,1060,455)
    sele=False
    chan=False
    sege=False
root.filename="D:/My Study/7th Semester/Major Project/square.jpg"
path="D:/My Study/7th Semester/Major Project/square.jpg"       
imshow(path,40,130)
lblred=Label(root,text="RED")
lblred.place(x=340,y=120)
channelback(path,340,140)
lblgr=Label(root,text="GREEN")
lblgr.place(x=560,y=120)
channelback(path,560,140)
lblbl=Label(root,text="BLUE")
lblbl.place(x=780,y=120)
channelback(path,780,140)
lblcy=Label(root,text="CYAN")
lblcy.place(x=340,y=350)
channelback(path,340,370)
lblmg=Label(root,text="MAGENTA")
lblmg.place(x=560,y=350)
channelback(path,560,370)
lblyl=Label(root,text="YELLOW")
lblyl.place(x=780,y=350)
channelback(path,780,370)
lblhue=Label(root,text="HUE")
lblhue.place(x=340,y=580)
channelback(path,340,600)
lblval=Label(root,text="SATURATION")
lblval.place(x=560,y=580)
channelback(path,560,600)
lblsatu=Label(root,text="VALUE")
lblsatu.place(x=780,y=580)
channelback(path,780,600)
btn1=Button(root,text="Select File",bg="black",fg="white",command= click)
btn1.place(x=40,y=60)
lblt1=Label(root,text="Input Image")
lblt1.place(x=40,y=100)
btn2=Button(root,text="Channelize",bg="black",fg="white",command=channel)
btn2.place(x=340,y=60)
lablt2=Label(root,text="Channelized Images")
lablt2.place(x=340,y=100)
btn3=Button(root,text="Segmentation",bg="black",fg="white",command=stype)
btn3.place(x=1060,y=60)
btn4=Button(root,text="Reset Button",bg="black",fg="white",command=reset)
btn4.place(x=440,y=60)
#print gd_path
#print root.filename
btn5=tk.Button(root,text="Separation",bg="black",fg="white",command=lambda: createNewWindow(gd_path))
btn5.pack()
btn5.place(x=1060,y=730)
variable = StringVar(root)
variable.set("RED") 
w = OptionMenu(root, variable, "RED", "GREEN", "BLUE","CYAN", "MAGENTA", "YELLOW","HUE", "SATURATION", "VALUE")
w.pack()
w.place(x=1160,y=60)
lablt3=Label(root,text="Segmented Images")
lablt3.place(x=1060,y=100)
lablt4=Label(root,text="1.K-Means Clustering", font=('Helvetica', 11, 'bold'))
lablt4.place(x=1060,y=120)
imshow(path,1060,155)
lablt4=Label(root,text="2.Fuzzy C Mean Clustering", font=('Helvetica', 11, 'bold'))
lablt4.place(x=1060,y=420)
imshow(path,1060,455)
root.mainloop()
