import numpy as np

marks_paths=["/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/clean.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/cw,tar,LL.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/cw,tar,NEXT.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/cw,untar,k10.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/cw,untar,k20.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/cw,untar,k30.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/cw,untar,k40.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/FGSM,16.npy",
		"/home/nesa320/Ji_3160102420/magnet/MagNet-master/mark/IGSM,20.npy"]
labels_adv_paths=["/home/nesa320/Ji_3160102420/fgsm_new/results/slim-10000-labels.npy",
			"/home/nesa320/Ji_3160102420/fgsm_new/results/labels-Slim-min-10000.npy",
			"/home/nesa320/Ji_3160102420/fgsm_new/results/labels-Slim-sec-10000.npy",
			"/home/nesa320/Ji_3160102420/magnet/MagNet-master/results/labels-untarget-slim-CW-k10.npy",
			"/home/nesa320/Ji_3160102420/magnet/MagNet-master/results/labels-untarget-slim-CW-k20.npy",
			"/home/nesa320/Ji_3160102420/magnet/MagNet-master/results/labels-untarget-slim-CW-k30.npy",
			"/home/nesa320/Ji_3160102420/magnet/MagNet-master/results/labels-untarget-slim-CW-k40.npy",
			"/home/nesa320/Ji_3160102420/fgsm_new/results/labels-FGSM-16.0.npy",
			"/home/nesa320/Ji_3160102420/fgsm_new/results/labels-IGSM-20.npy"]
labels_ori_path="/home/nesa320/Ji_3160102420/fgsm_new/results/slim-10000-labels.npy"
names=["Ori","LL","NEXT","Untarget 10","Untarget 20","Untarget 30","Untarget 40","FGSM 16","IGSM 20"]
bounds=[0.002,0.0022,0.0025,0.0028,0.003,0.0032,0.004,0.005]
for bound in bounds:
	for i in range(len(marks_paths)):
		mark_now=np.load(marks_paths[i])
		labels_adv=np.load(labels_adv_paths[i])
		labels_ori=np.load(labels_ori_path)
		all_count=0
		adv_count=0
		for j in range(mark_now.shape[0]):
			if(mark_now[j]>=bound):
				all_count=all_count+1
				if(labels_adv[j]!=labels_ori[j]):
					adv_count=adv_count+1
		f=open("mark_result.txt","a+")
		f.write(names[i]+" "+str(bound)+" :"+str(all_count)+",")
		f.write("adv:"+str(adv_count)+"\n")
		f.close()