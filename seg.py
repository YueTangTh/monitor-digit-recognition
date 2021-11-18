import numpy as np
import cv2
import os

mat10_1 = [[616, 657, 418, 490], [638, 657, 694, 723], [516, 543, 1728, 1768],
    [604, 644, 1585, 1630], [604, 647, 1637, 1686], [616, 656, 1714, 1764],
    [748, 790, 1565, 1610], [757, 796, 1621, 1664], [767, 805, 1694, 1740],
    [820, 859, 1541, 1604], [888, 928, 1532, 1594]]

mat10_2 = [[613, 655, 443, 510], [638, 657, 713, 747], [516, 543, 1755, 1797],
    [604, 644, 1612, 1657], [608, 649, 1664, 1713], [616, 659, 1741, 1792],
    [750, 790, 1592, 1635], [758, 798, 1646, 1690], [767, 805, 1720, 1769],
    [821, 861, 1566, 1630], [888, 928, 1556, 1621]]

mat10_3 = [[600, 643, 453, 514], [621, 642, 713, 747], [496, 523, 1695, 1737], 
    [576, 618, 1566, 1607], [584, 624, 1614, 1657], [586, 629, 1684, 1730],
    [718, 752, 1547, 1588], [723, 758, 1598, 1640], [727, 769, 1667, 1708], 
    [785, 821, 1520, 1585], [848, 887, 1506, 1576]]

mat10_4 = [[600, 643, 447, 507], [621, 642, 706, 740], [496, 523, 1689, 1731], 
    [576, 618, 1560, 1602], [584, 624, 1608, 1651], [586, 629, 1678, 1724],
    [718, 752, 1541, 1583], [723, 758, 1592, 1634], [732, 769, 1661, 1702], 
    [784, 821, 1517, 1580], [846, 887, 1506, 1570]]

mat11_1 = [[649, 681, 490, 554],[663, 680, 758, 784],[522, 547, 1742, 1782],
    [607, 644, 1610, 1654], [610, 647, 1662, 1705], [614, 653, 1734, 1779],
    [745, 783, 1598, 1640], [753, 788, 1650, 1692], [758, 795, 1720, 1762],
    [813, 849, 1574, 1616], [880, 920, 1569, 1628]]

#mat11_2
h_shift = -2
w_shift = -6
mat11_2 = mat11_1
for i in range(len(mat10_1)):
    for j in range(2):
        mat11_2[i][j] = mat11_1[i][j] + h_shift
        mat11_2[i][j+2] = mat11_2[i][j+2] + w_shift

mat12_1 = [[635, 667, 468, 534], [661, 676, 753, 777], [520, 539, 1736, 1774],
    [604, 643, 1603, 1645], [605, 643, 1655, 1697], [610, 650, 1728, 1772],
    [743, 780, 1590, 1630], [748, 783, 1640, 1684], [753, 792, 1712, 1755],
    [811, 846, 1566, 1629], [876, 914, 1560, 1620]]

#mat12_2
h_shift = -1
w_shift = -6
mat12_2 = mat12_1
for i in range(len(mat10_1)):
    for j in range(2):
        mat12_2[i][j] = mat12_1[i][j] + h_shift
        mat12_2[i][j+2] = mat12_1[i][j+2] + w_shift

#mat12_3
h_shift = -1
w_shift = -2
mat12_3 = mat12_1
for i in range(len(mat10_1)):
    for j in range(2):
        mat12_3[i][j] = mat12_1[i][j] + h_shift
        mat12_3[i][j+2] = mat12_1[i][j+2] + w_shift

#mat12_4
mat12_4 = [[650, 688, 303, 373], [673, 693, 602, 633], [522, 544, 1545, 1581],
    [602, 642, 1430, 1468], [605, 642, 1475, 1520], [608, 648, 1543, 1584],
    [738, 775, 1420, 1458], [743, 778, 1468, 1507], [746, 781, 1530, 1570],
    [806, 841, 1398, 1455], [865, 903, 1393, 1455]]

#mat12_5
mat12_5 = [[635, 672, 523, 584], [665, 685, 800, 823], [530, 554, 1802, 1844],
    [614, 654, 1665, 1710], [620, 658, 1720, 1762], [625, 665, 1793, 1842],
    [758, 795, 1652, 1693], [763, 800, 1705, 1750], [770, 809, 1776, 1823],
    [826, 861, 1626, 1686], [893, 932, 1621, 1685]]

mat13_1 =[[639, 669, 533, 583], [665, 681, 794, 821], [528, 553, 1796, 1839],
    [614, 652, 1662, 1705], [619, 656, 1712, 1759], [624, 659, 1788, 1833],
    [755, 796, 1647, 1688], [762, 800, 1697, 1745], [769, 806, 1771, 1817],
    [822, 862, 1620, 1663], [891, 931, 1615, 1680]]

# Define the codec create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
mat = mat11_1

video_path = []
image_path = []
for k in range(len(mat)):
    video_path.append('/home/yuetang/OCR/OTSU/%d' % (k+1) + '.avi')

for j in range(1):
    h_upper = mat[j][1]
    h_lower = mat[j][0]
    w_upper = mat[j][3]
    w_lower = mat[j][2]
    height = h_upper - h_lower
    width = w_upper - w_lower
    out_cropped = cv2.VideoWriter(video_path[j], fourcc, 3.0, (width, height))

    frame_num = 0
    count = 0
    cap = cv2.VideoCapture('/home/yuetang/OCR/OTSU/test/video/20190614_113747.avi')
    while(cap.isOpened() and count < 1800):
        ret, frame = cap.read()
        if ret == True:
          frame_num = frame_num +1
          if frame_num == 10:
            img_cropped = frame[h_lower:h_upper, w_lower:w_upper]
            out_cropped.write(img_cropped)
            count = count + 1
            frame_num = 0
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
          break
    out_cropped.release()
    cap.release()
    print('Finished %d'%j)

'''  
# Define the codec create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
mat = mat11_1

video_path = []
image_path = []
for k in range(len(mat)):
    video_path.append('tests/roi/%d' % (k+1) + '.avi')

for j in range(1):
    h_upper = mat[j][1]
    h_lower = mat[j][0]
    w_upper = mat[j][3]
    w_lower = mat[j][2]
    height = h_upper - h_lower
    width = w_upper - w_lower
    out_cropped = cv2.VideoWriter(video_path[j], fourcc, 30.0, (width, height))

    frame_num = 0
    count = 0
    cap = cv2.VideoCapture('tests/videos/20190614_113747.avi')
    while(cap.isOpened() and count < 180):
        ret, frame = cap.read()
        if ret == True:
          frame_num = frame_num +1
          if frame_num == 1:
            img_cropped = frame[h_lower:h_upper, w_lower:w_upper]
            out_cropped.write(img_cropped)
            count = count + 1
            frame_num = 0
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
          break
    out_cropped.release()
    cap.release()
    print('Finished %d'%j)

#cap.release()
#cv2.destroyAllWindows()'''