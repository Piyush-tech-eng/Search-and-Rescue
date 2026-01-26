# import cv2 as cv
import first as f
import second as s
import math
res_pad=[]
sub=[]
for i in range(len(f.center_arr)):
    if f.center_arr[i][0]=='Circle':
        if f.center_arr[i][1]=='Blue':
            res_pad.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Pink':
            res_pad.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Gray':
            res_pad.append(f.center_arr[i])
    elif f.center_arr[i][0]=='Star':
        if f.center_arr[i][1]=='Red':
            sub.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Yellow':
            sub.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Green':
            sub.append(f.center_arr[i])
    elif f.center_arr[i][0]=='Triangle':
        if f.center_arr[i][1]=='Red':
            sub.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Yellow':
            sub.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Green':
            sub.append(f.center_arr[i])
    elif f.center_arr[i][0]=='Square':
        if f.center_arr[i][1]=='Red':
            sub.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Yellow':
            sub.append(f.center_arr[i])
        elif f.center_arr[i][1]=='Green':
            sub.append(f.center_arr[i])
for i in sub:
    if i[0]=='Star':
        if i[1]=='Red':
            i.append(9)
        elif i[1]=='Yellow':
            i.append(6)
        elif i[1]=='Green':
            i.append(4)
    elif i[0]=='Triangle':
        if i[1]=='Red':
            i.append(6)
        elif i[1]=='Yellow':
            i.append(4)
        elif i[1]=='Green':
            i.append(3)
    elif i[0]=='Square':
        if i[1]=='Red':
            i.append(3)
        elif i[1]=='Yellow':
            i.append(2)
        elif i[1]=='Green':
            i.append(1)
max_p = max(x[4] for x in sub)
prt=[]
for i in sub:
    i.append((max_p-i[4]+1)/max_p)
dist=[]
score=[]
for i in range(0,3):
    for j in range(len(sub)):
            dist.append([res_pad[i][1],sub[j][0],sub[j][1],math.dist([res_pad[i][2],res_pad[i][3]],[sub[j][2],sub[j][3]])])
            score.append([res_pad[i][1],sub[j][0],sub[j][1]])
for i in range(0, 3):
    d = i * len(sub)
    di = (i + 1) * len(sub)

    min_d = min(x[3] for x in dist[d:di])
    max_d = max(x[3] for x in dist[d:di])

    for j in range(len(sub)):
        a = dist[d + j][3] - min_d + 30
        b = max_d - min_d
        if b != 0:  # avoid division by zero
            dist[j+((i)*len(sub))].append(1 - (a / b))
        else:
            dist[j+((i)*len(sub))].append(0)
alpha=0.6
lambda_=0.1
for i in range(len(dist)):
    if dist[i][0]=='Blue':
        for j in range(len(sub)):
            if sub[j][0]==dist[i][1] and sub[j][1]==dist[i][2]:
                score[i].append(sub[j][5] * math.exp(-lambda_ * dist[i][4]))
                '''score[i].append((alpha*sub[j][5])+((1-alpha)*dist[i][4]))'''
    elif dist[i][0]=='Pink':
        for j in range(len(sub)):
            if sub[j][0]==dist[i][1] and sub[j][1]==dist[i][2]:
                score[i].append(sub[j][5] * math.exp(-lambda_ * dist[i][4]))
                ''' score[i].append((alpha*sub[j][5])+((1-alpha)*dist[i][4]))'''
    elif dist[i][0]=='Gray':
        for j in range(len(sub)):
            if sub[j][0]==dist[i][1] and sub[j][1]==dist[i][2]:
                score[i].append(sub[j][5] * math.exp(-lambda_ * dist[i][4]))
                '''score[i].append((alpha*sub[j][5])+((1-alpha)*dist[i][4]))'''



print(score)

# -----------------------------
# Display
# -----------------------------
# cv.imshow("Original", f.output)
# cv.imshow("Land Mask (Clean)", s.land_mask)
# cv.imshow("Land vs Ocean Overlay", s.output)
# print(f.center_arr)
# cv.imwrite("land_ocean_final.png", s.output)
# cv.imshow("Original_hsv", f.hsv)
# cv.waitKey(0)
# cv.destroyAllWindows()
