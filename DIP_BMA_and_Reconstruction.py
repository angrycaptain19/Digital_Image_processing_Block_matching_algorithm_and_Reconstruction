import numpy as np
from numpy import pi, exp, sqrt
import glob
import cv2
import math
import matplotlib.pyplot as plt
from random import randint
import timeit
import os.path
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
import queue
def main():
    img1 = cv2.imread('FOREMAN000.tif', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('FOREMAN001.tif', cv2.IMREAD_GRAYSCALE)
    img1 = img1[:152,:]
    img2 = img2[:152,:]
    img3 = cv2.imread('FOREMAN072.tif', cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread('FOREMAN073.tif', cv2.IMREAD_GRAYSCALE)
    Vector2 = Three_Step_Search(img2, img1, 8, 4)
    Vector = Simple_and_Efficient_search(img2,img1,8)
    #Vector = EBMA(img2, img1, 8,4,1)
    Vector2 = Diamond_Search(img2,img1,8)
    #vector_quiver(Vector)
    Reconstructed = Reconstruction(img1, Vector, 8)
    cv2.waitKey(0)
    img_re = cv2.imread('SE_search,8.tif',cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('EBMA_Thresh200,8-4.tif',Reconstructed)

    PSNR(img2,Reconstructed)
    Vector = EBMA(img2, img1, 8 , 8, 1)
    vector_quiver(Vector)
    Vector1 = EBMA(img3, img2, 16, 8, 1)
    Vector2 = EBMA(img4, img3, 16, 8, 1)

    img_t = EBMA_Reconstruction(img1,Vector,8)
    img_tt = EBMA_Reconstruction(img_t,Vector1,16)
    img_ttt = EBMA_Reconstruction(img_tt,Vector2,16)

    cv2.imshow('imgt',img_t)
    cv2.waitKey(0)
    cv2.imwrite('Reconstructed_071.tif', img_t)
    cv2.imwrite('Reconstructed_072.tif', img_tt)
    cv2.imwrite('Reconstructed_073.tif',img_ttt)
    Vector1 = Log_search(img2, img1, 16, 8)
    cv2.imshow('imgt',Reconstruction(img2,Vector1,16))
    cv2.waitKey(0)


    cv2.imshow('imgt',Reconstruction(img2,Vector2,16))
    cv2.waitKey(0)

    vector_quiver(Vector)
    vector_quiver(Vector1)
    vector_quiver(Vector2)
    return

def Huffman_encoding():
    class Node:
        def __init__(self):
            self.prob = None
            self.code = None
            self.data = None
            self.left = None
            self.right = None  # the color (the bin value) is only required in the leaves

        def __lt__(self, other):
            if (self.prob < other.prob):  # define rich comparison methods for sorting in the priority queue
                return 1
            else:
                return 0

        def __ge__(self, other):
            if (self.prob > other.prob):
                return 1
            else:
                return 0

    def rgb2gray(img):
        gray_img = np.rint(img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1140)
        gray_img = gray_img.astype(int)
        return gray_img

    def get2smallest(data):  # can be used instead of inbuilt function get(). was not used in  implementation
        first = second = 1;
        fid = sid = 0
        for idx, element in enumerate(data):
            if (element < first):
                second = first
                sid = fid
                first = element
                fid = idx
            elif (element < second and element != first):
                second = element
        return fid, first, sid, second

    def tree(probabilities):
        prq = queue.PriorityQueue()
        for color, probability in enumerate(probabilities):
            leaf = Node()
            leaf.data = color
            leaf.prob = probability
            prq.put(leaf)

        while (prq.qsize() > 1):
            newnode = Node()  # create new node
            l = prq.get()
            r = prq.get()  # get the smalles probs in the leaves
            # remove the smallest two leaves
            newnode.left = l  # left is smaller
            newnode.right = r
            newprob = l.prob + r.prob  # the new prob in the new node must be the sum of the other two
            newnode.prob = newprob
            prq.put(newnode)  # new node is inserted as a leaf, replacing the other two
        return prq.get()  # return the root node - tree is complete

    def huffman_traversal(root_node, tmp_array, f):  # traversal of the tree to generate codes
        if (root_node.left is not None):
            tmp_array[huffman_traversal.count] = 1
            huffman_traversal.count += 1
            huffman_traversal(root_node.left, tmp_array, f)
            huffman_traversal.count -= 1
        if (root_node.right is not None):
            tmp_array[huffman_traversal.count] = 0
            huffman_traversal.count += 1
            huffman_traversal(root_node.right, tmp_array, f)
            huffman_traversal.count -= 1
        else:
            huffman_traversal.output_bits[
                root_node.data] = huffman_traversal.count  # count the number of bits for each color
            bitstream = ''.join(str(cell) for cell in tmp_array[1:huffman_traversal.count])
            color = str(root_node.data)
            wr_str = color + ' ' + bitstream + '\n'
            f.write(wr_str)  # write the color and the code to a file
        return

    # Read an bmp image into a numpy array
    #img = imresize(img, 10)  # resize to 10% (not strictly necessary - done for faster computation)
    img = cv2.imread('Lenna.bmp', cv2.IMREAD_GRAYSCALE)

    # compute histogram of pixels
    hist = np.bincount(img.ravel(), minlength=256)

    probabilities = hist / np.sum(hist)  # a priori probabilities from frequencies

    root_node = tree(probabilities)  # create the tree using the probs.
    tmp_array = np.ones([64], dtype=int)
    huffman_traversal.output_bits = np.empty(256, dtype=int)
    huffman_traversal.count = 0
    f = open('codes.txt', 'w')
    huffman_traversal(root_node, tmp_array, f)  # traverse the tree and write the codes

    input_bits = img.shape[0] * img.shape[1] * 8  # calculate number of bits in grayscale
    compressed_bit = np.sum(huffman_traversal.output_bits * hist)
    compression = (1 - np.sum(huffman_traversal.output_bits * hist) / input_bits) * 100  # compression rate
    print(input_bits, compressed_bit)
    print(np.max(huffman_traversal.output_bits))
    print(hist)
    print('Compression is ', compression, ' percent')

    return


def EBMA(imgt,imgt_1,block_size,search_range,step_size):
###### imgt = anchor frame, imgt_1 = target frame #####

## step_size =1 : integer_pel accuracy search
    y,x = imgt.shape
    #print(imgt.shape,imgt_1.shape)
    imgt = np.array(imgt, dtype = np.uint8)
    imgt_1 = np.array(imgt_1,dtype = np.uint8)
    Search_range= 2*search_range + 1
    SAD = np.zeros((Search_range,Search_range))
    Motion_vector = []
    Motion_vector = np.array(Motion_vector, dtype='int8')
    sum = 0
    half_size = int(block_size/2)
    for i in range(half_size,y,block_size):
        for j in range(half_size,x,block_size):
            Frame_t = np.array(imgt[i-half_size:i+half_size,j-half_size:j+half_size],dtype ='int16')
            p = int(y/block_size)
            q = int(x/block_size)
            for z in range(i-search_range,i+search_range+1,step_size):
                for v in range(j-search_range,j+search_range+1,step_size):
                    Frame_t_1 = np.array(imgt_1[z-half_size:z+half_size,v-half_size:v+half_size],dtype = 'int16')
                    #Sub = Frame_t - Frame_t_1
                    #print(i,j,z,v)
                    if Frame_t_1.shape != (block_size,block_size):
                        SAD[z-i+search_range][v-j+search_range] = float('inf')
                    else:
                        SAD[z-i+search_range][v-j+search_range] = np.sum(np.absolute(Frame_t - Frame_t_1))
                        sum += 1

            a = np.argwhere(SAD==np.min(SAD))-search_range
            if np.min(SAD) > 100:
                Motion_vector =np.append(Motion_vector, (a[0][0],a[0][1]))
            else :
                Motion_vector = np.append(Motion_vector,(0,0))
    Motion_vector = Motion_vector.reshape(p,q,2)
    print(sum)
    return Motion_vector

def Log_search (imgt,imgt_1,block_size,search_range):

    y,x = imgt.shape
    #print(imgt.shape,imgt_1.shape)
    imgt = np.array(imgt, dtype = np.uint8)
    imgt_1 = np.array(imgt_1,dtype = np.uint8)

    SAD = np.zeros(5)
    SAD = np.array(SAD,dtype ='float')
    Motion_vector = []
    Motion_vector = np.array(Motion_vector, dtype='int16')
    half_size = int(block_size/2)
    p = int(y / block_size)
    q = int(x / block_size)
    sum = 0

    for i in range(half_size,y,block_size):
        for j in range(half_size,x,block_size):
            Frame_t = np.array(imgt[i-half_size:i+half_size,j-half_size:j+half_size],dtype ='int16')

            i_o = i
            j_o = j
            half_range = int(search_range/2)

            while half_range > 0.5 :

                    Frame_t_1_block = np.array(imgt_1[i-half_range-half_size:i-half_range+half_size,j-half_size:j+half_size],dtype = 'int16')
                    if Frame_t_1_block.shape == (block_size,block_size):
                        SAD[0] = np.sum(np.absolute(Frame_t - Frame_t_1_block))
                        sum += 1
                    else :
                        SAD[0] = 'inf'

                    Frame_t_1_block = np.array(imgt_1[i-half_size:i+half_size,j-half_range-half_size:j-half_range+half_size], dtype = 'int16')
                    if Frame_t_1_block.shape == (block_size, block_size):
                        SAD[1] = np.sum(np.absolute(Frame_t - Frame_t_1_block))
                        sum += 1
                    else:
                        SAD[1] = 'inf'

                    Frame_t_1_block = np.array(imgt_1[i-half_size: i+half_size, j-half_size:j+half_size],dtype = 'int16')
                    SAD[2] = np.sum(np.absolute(Frame_t - Frame_t_1_block))
                    sum += 1

                    Frame_t_1_block = np.array(imgt_1[i-half_size:i+half_size,j+half_range-half_size:j+half_range+half_size], dtype = 'int16')
                    if Frame_t_1_block.shape == (block_size, block_size):
                        SAD[3] = np.sum(np.absolute(Frame_t - Frame_t_1_block))
                        sum += 1
                    else:
                        SAD[3] = 'inf'
                    Frame_t_1_block = np.array(imgt_1[i+half_range-half_size:i+half_range+half_size,j-half_size:j+half_size],dtype = 'int16')
                    if Frame_t_1_block.shape == (block_size, block_size):
                        SAD[4] = np.sum(np.absolute(Frame_t - Frame_t_1_block))
                        sum += 1
                    else:
                        SAD[4] = 'inf'

                    min = np.argmin(SAD)

                    if min == 2 :
                       half_range = int(half_range/2)
                    elif min == 0 :
                        i = i - half_range
                    elif min == 1 :
                        j = j - half_range
                    elif min == 3 :
                        j = j + half_range
                    elif min == 4 :
                        i = i + half_range

            Motion_vector = np.append(Motion_vector,(i-i_o,j-j_o))
            i = i_o
            j = j_o
    Motion_vector = Motion_vector.reshape(p, q, 2)
    print(sum)
    return  Motion_vector

def Three_Step_Search(imgt,imgt_1,block_size,search_range):
    y, x = imgt.shape
    # print(imgt.shape,imgt_1.shape)
    imgt = np.array(imgt, dtype=np.uint8)
    imgt_1 = np.array(imgt_1, dtype=np.uint8)
    SAD = np.zeros(9)
    SAD = np.array(SAD, dtype='float')
    Motion_Vector = []
    Motion_Vector = np.array(Motion_Vector,dtype = 'int16')
    Motion_vector = np.zeros((9,2))
    Motion_vector = np.array(Motion_vector, dtype='int16')
    half_size = int(block_size / 2)
    p = int(y / block_size)
    q = int(x / block_size)
    sum = 0
    for i in range(half_size, y, block_size):
        for j in range(half_size, x, block_size):
            Frame_t = np.array(imgt[i - half_size:i + half_size, j - half_size:j + half_size], dtype='int16')
            i_o = i
            j_o = j

            half_range = int(search_range/2)
            while half_range > 0.5:
                index = 0
                for u in range( i-half_range,i+half_range+1,half_range):
                   for v in range( j-half_range,j+half_range+1,half_range):
                        Frame_t_1 = np.array(imgt_1[u-half_size:u+half_size,v-half_size:v+half_size],dtype ='int16')
                        if Frame_t_1.shape != (block_size, block_size):
                            SAD[index] = float('inf')
                            Motion_vector[index] = [0,0]
                        else:
                            Sub = Frame_t - Frame_t_1
                            SAD[index] = np.sum(np.absolute(Frame_t - Frame_t_1))
                            sum += 1
                            Motion_vector[index] = [u-i,v-j]
                        index += 1
                min = np.argmin(SAD)
                i = i + Motion_vector[min][0]
                j = j + Motion_vector[min][1]
                half_range = int(half_range/2)
            Motion_Vector = np.append(Motion_Vector,(i-i_o,j-j_o))
            i = i_o
            j = j_o

    Motion_Vector = Motion_Vector.reshape(p, q, 2)
    print(sum)
    return Motion_Vector

def Diamond_Search(imgt,imgt_1,block_size):

    y, x = imgt.shape
    imgt = np.array(imgt, dtype = np.uint8)
    imgt_1 = np.array(imgt_1, dtype = np.uint8)
    SAD_1 = np.zeros(9)
    SAD_1 = np.array(SAD_1, dtype = 'float')
    SAD_temp = np.zeros(9)
    SAD_temp = np.array(SAD_temp, dtype = 'float')

    Index_array_1 = np.zeros((9,2))
    Index_array_1 = np.array(Index_array_1, dtype = 'int16')
    Index_array_2 = np.zeros((5,2))
    Index_array_2 = np.array(Index_array_2, dtype='int16')
    SAD_2 = np.zeros(5)
    SAD_2 = np.array(SAD_2, dtype = 'float')


    Motion_Vector = []
    Motion_Vector = np.array(Motion_Vector, dtype='int16')

    Motion_vector = np.zeros((9, 2))
    Motion_vector = np.array(Motion_vector, dtype='int16')

    half_size = int(block_size / 2)
    p = int(y / block_size)
    q = int(x / block_size)
    sum = 0
    for i in range(half_size, y, block_size):
        for j in range(half_size, x, block_size):
            Frame_t = np.array(imgt[i - half_size:i + half_size, j - half_size:j + half_size], dtype='int16')
            i_o = i
            j_o = j
            min1 = 0

            while min1 != 4:
                index_1 = 0

                for u in range(i-2,i+3):
                    for v in range(j-2,j+3):
                        overlap= 100
                        if (
                            u + v in [i + j - 2, i + j, i + j + 2]
                            and (u, v) != (i - 2, j + 2)
                            and (u, v) != (i + 2, j - 2)
                        ):

                            for k in range(9):
                                if Index_array_1[k][0] == u and Index_array_1[k][1] == v :
                                    overlap = k

                            if overlap == 100:
                                Frame_t_1 = np.array(imgt_1[u-half_size:u+half_size,v-half_size:v+half_size],dtype ='int16')
                                Index_array_1[index_1] = [u,v]
                                if Frame_t_1.shape == (block_size, block_size):
                                       #Sub = Frame_t - Frame_t_1
                                    SAD_1[index_1] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                    SAD_temp[index_1] = SAD_1[index_1]
                                    sum += 1
                                else:
                                    SAD_1[index_1] = float('inf')
                                    SAD_temp[index_1] = float('inf')
                            else:
                                SAD_1[index_1] = SAD_temp[overlap]
                                Index_array_1[index_1] = [u, v]

                            index_1 += 1

                min1 = np.argmin(SAD_1)
                i = Index_array_1[min1][0]
                j = Index_array_1[min1][1]

            index_2 = 0
            for t in range(i-1,i+2):
                for s in range(j-1,j+2):
                    if t+s == i+j-1 or t+s == i+j+1 or (t,s) == (i,j):
                        Frame_t_1 = np.array(imgt_1[t-half_size:t+half_size,s-half_size:s+half_size],dtype = 'int16')
                        Index_array_2[index_2] = [t,s]
                        if Frame_t_1.shape == (block_size, block_size):
                            SAD_2[index_2] = np.sum(np.absolute(Frame_t - Frame_t_1))
                            sum += 1
                        else:
                            SAD_2[index_2] = float('inf')
                        index_2 +=1
            min2 = np.argmin(SAD_2)
            i = Index_array_2[min2][0]
            j = Index_array_2[min2][1]

            Motion_Vector = np.append(Motion_Vector, (i-i_o,j-j_o))
            i = i_o
            j = j_o
    Motion_Vector = Motion_Vector.reshape(p, q, 2)
    print(sum)
    return Motion_Vector

def Simple_and_Efficient_search (imgt,imgt_1,block_size):
    y, x = imgt.shape
    print(imgt.shape,imgt_1.shape)
    imgt = np.array(imgt, dtype=np.uint8)
    imgt_1 = np.array(imgt_1, dtype=np.uint8)
    MAD = np.zeros(4)
    MAD = np.array(MAD, dtype='float')

    Index_array = np.zeros((4,2))
    Index_array = np.array(Index_array,dtype = 'int16')

    Motion_vector = np.zeros((9,2))
    Motion_vector = np.array(Motion_vector, dtype='int16')
    SAD = np.zeros(9)
    SAD = np.array(SAD, dtype='float')

    Motion_Vector = []
    Motion_Vector = np.array(Motion_Vector,dtype = 'int16')

    half_size = int(block_size / 2)
    p = int(y / block_size)
    q = int(x / block_size)
    sum =  0
    for i in range(half_size, y, block_size):
        for j in range(half_size, x, block_size):
            Frame_t = np.array(imgt[i - half_size:i + half_size, j - half_size:j + half_size], dtype='int16')
            i_o = i
            j_o = j

            search_range = 4
            #print('c')
            if i == half_size or j == half_size or i == y - half_size or j == x - half_size:
                Motion_Vector = np.append(Motion_Vector,(0,0))

                #print(sum)
            else:

                while search_range > 0.5:
                    Frame_t_1_5 = np.array(imgt_1[i-half_size:i+half_size,j-half_size:j+half_size],dtype ='int16')
                    Frame_t_1_6 = np.array(imgt_1[i-half_size:i+half_size,j+search_range-half_size:j+search_range+half_size],dtype ='int16')
                    Frame_t_1_8 = np.array(imgt_1[i+search_range-half_size:i+search_range+half_size,j-half_size:j+half_size],dtype ='int16')

                    MAD5 = np.sum(np.absolute(Frame_t - Frame_t_1_5))
                    MAD6 = np.sum(np.absolute(Frame_t - Frame_t_1_6))
                    MAD8 = np.sum(np.absolute(Frame_t - Frame_t_1_8))
                    MAD[0] = MAD5
                    MAD[1] = MAD6
                    MAD[2] = MAD8
                    Index_array[0] = [ i,j ]
                    Index_array[1] = [ i, j+search_range]
                    Index_array[2] = [ i+search_range,j]

                    if MAD6 < MAD5 and MAD8 < MAD5:
                        Frame_t_1_check = np.array(imgt_1[i+search_range-half_size:i+search_range+half_size,j+search_range-half_size:j+search_range+half_size],dtype ='int16')
                        MAD_check = np.sum(np.absolute(Frame_t - Frame_t_1_check))
                        sum += 1
                        MAD[3] = MAD_check
                        Index_array[3] = [i+search_range, j+search_range]
                        min = np.argmin(MAD)
                        i = Index_array[min][0]
                        j = Index_array[min][1]
                        search_range = int(search_range/2)
                        #print('1')
                    elif MAD6 < MAD5 < MAD8 :
                        Frame_t_1_check = np.array(imgt_1[i - search_range - half_size:i - search_range + half_size, j + search_range - half_size:j + search_range + half_size],dtype='int16')
                        MAD_check = np.sum(np.absolute(Frame_t - Frame_t_1_check))
                        sum += 1
                        MAD[3] = MAD_check
                        Index_array[3] = [i - search_range, j + search_range]
                        min = np.argmin(MAD)
                        i = Index_array[min][0]
                        j = Index_array[min][1]
                        search_range = int(search_range/2)
                        #print('2')
                    elif MAD8 < MAD5 < MAD6 :
                        Frame_t_1_check = np.array(imgt_1[i + search_range - half_size:i + search_range + half_size, j - search_range - half_size:j - search_range + half_size],dtype='int16')
                        MAD_check = np.sum(np.absolute(Frame_t - Frame_t_1_check))
                        sum += 1
                        MAD[3] = MAD_check
                        Index_array[3] = [i + search_range, j - search_range]
                        min = np.argmin(MAD)
                        i = Index_array[min][0]
                        j = Index_array[min][1]
                        search_range = int(search_range/2)
                        #print('3')
                    elif MAD5 < MAD6 and MAD5 < MAD8 :

                        if MAD5 < MAD6 < MAD8 :

                            Frame_t_1_check = np.array(imgt_1[i - search_range - half_size:i - search_range + half_size,j - half_size:j + half_size],dtype='int16')
                            MAD_check = np.sum(np.absolute(Frame_t - Frame_t_1_check))
                            sum += 1
                            if MAD_check < MAD5:
                                    Frame_t_1_check = np.array(imgt_1[i - search_range - half_size : i - search_range + half_size, j - search_range -half_size : j - search_range +half_size],dtype = 'int16')
                                    MAD_check_1 = np.sum(np.absolute(Frame_t - Frame_t_1_check))
                                    sum += 1
                                    if MAD_check > MAD_check_1 :
                                        i = i - search_range
                                        j = j - search_range
                                        search_range = int(search_range/2)
                                    elif MAD_check_1 >= MAD_check :
                                        i = i - search_range
                                        search_range = int(search_range/2)
                            elif MAD_check >= MAD5:

                                    search_range = int(search_range/2)
                                    #print(search_range)

                        elif MAD5 < MAD8 < MAD6 :
                            #print('5')
                            Frame_t_1_check = np.array(imgt_1[i - half_size:i + half_size,j- search_range - half_size:j -search_range + half_size],dtype='int16')
                            MAD_check = np.sum(np.absolute(Frame_t - Frame_t_1_check))
                            sum += 1
                            if MAD_check < MAD5:
                                    Frame_t_1_check = np.array(imgt_1[i - search_range - half_size : i - search_range + half_size, j - search_range -half_size : j - search_range +half_size],dtype = 'int16')
                                    MAD_check_1 = np.sum(np.absolute(Frame_t - Frame_t_1_check))
                                    sum += 1
                                    if MAD_check > MAD_check_1 :
                                        i = i - search_range
                                        j = j - search_range
                                        search_range = int(search_range/2)
                                    elif MAD_check_1 >= MAD_check :
                                        j = j - search_range
                                        search_range = int(search_range/2)
                            elif MAD_check >= MAD5:
                                    search_range = int(search_range/2)
                        elif MAD6 == MAD8 :
                            index = 0

                            for u in range(i - search_range, i + search_range + 1, search_range):
                                for v in range(j - search_range, j + search_range + 1, search_range):
                                    Frame_t_1 = np.array(
                                        imgt_1[u - half_size:u + half_size, v - half_size:v + half_size], dtype='int16')
                                    if Frame_t_1.shape != (block_size, block_size):
                                        SAD[index] = float('inf')
                                        Motion_vector[index] = [0, 0]

                                    else:
                                        SAD[index] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                        sum += 1
                                        Motion_vector[index] = [u - i, v - j]
                                    index += 1
                            min = np.argmin(SAD)
                            i = i + Motion_vector[min][0]
                            j = j + Motion_vector[min][1]
                            search_range = int(search_range / 2)

                    elif MAD5 == MAD6 or MAD5 == MAD8 or MAD6 == MAD8:
                        index = 0

                        for u in range(i - search_range, i + search_range + 1, search_range):
                            for v in range(j - search_range, j + search_range + 1, search_range):
                                Frame_t_1 = np.array(imgt_1[u - half_size:u + half_size, v - half_size:v + half_size],dtype='int16')
                                if Frame_t_1.shape != (block_size, block_size):
                                    SAD[index] = float('inf')
                                    Motion_vector[index] = [0, 0]

                                else:
                                    SAD[index] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                    sum += 1
                                    Motion_vector[index] = [u - i, v - j]
                                index += 1
                        min = np.argmin(SAD)
                        i = i + Motion_vector[min][0]
                        j = j + Motion_vector[min][1]
                        search_range = int(search_range / 2)


                Motion_Vector = np.append(Motion_Vector,(i-i_o,j-j_o))
                i = i_o
                j = j_o

    Motion_Vector = Motion_Vector.reshape(p, q, 2)
    print(sum)
    return Motion_Vector

def New_Three_step_Search(imgt,imgt_1,block_size):
    y, x = imgt.shape
    imgt = np.array(imgt, dtype=np.uint8)
    imgt_1 = np.array(imgt_1, dtype=np.uint8)

    SAD_1 = np.zeros(17)
    SAD_2 = np.zeros(9)

    SAD_1 = np.array(SAD_1, dtype='float')
    SAD_2 = np.array(SAD_2, dtype='float')

    Motion_Vector = []
    Motion_Vector = np.array(Motion_Vector, dtype='int16')

    Index_array_1 = np.zeros((17,2))
    Index_array_1 = np.array(Index_array_1, dtype='int16')

    Index_array_2 = np.zeros((9,2))
    Index_array_2 = np.array(Index_array_2, dtype='int16')

    half_size = int(block_size / 2)
    p = int(y / block_size)
    q = int(x / block_size)
    sum = 0
    for i in range(half_size, y, block_size):
        i_o = i
        for j in range(half_size, x, block_size):
            Frame_t = np.array(imgt[i - half_size:i + half_size, j - half_size:j + half_size], dtype='int16')
            j_o = j
            index = 0
            search_range = 4
            print(i, j_o)
            for u in range(i - 1, i + 2):
                for v in range(j_o - 1, j_o + 2):
                    Frame_t_1 = np.array(imgt_1[u - half_size:u + half_size, v - half_size:v + half_size],dtype='int16')
                    Index_array_1[index] = [u,v]
                    if Frame_t_1.shape != (block_size, block_size):
                        SAD_1[index] = float('inf')
                    else:
                        SAD_1[index] = np.sum(np.absolute(Frame_t - Frame_t_1))
                        sum += 1
                    index += 1

            for u in range(i-4,i+5,4):
                for v in range(j_o - 4, j_o + 5, 4):
                    if (u, v) != (i, j_o):
                        Frame_t_1 = np.array(imgt_1[u - half_size: u+half_size, v-half_size:v+half_size],dtype = 'int16')
                        Index_array_1[index] = [u, v]
                        if Frame_t_1.shape == (block_size, block_size):
                            SAD_1[index] = np.sum(np.absolute(Frame_t - Frame_t_1))
                            sum += 1
                        else:
                            SAD_1[index] = float('inf')
                        index += 1

            min = np.argmin(SAD_1)

            if min == 4:
                Motion_Vector = np.append(Motion_Vector,(0,0))

            else:
                if min <= 8:
                    u = Index_array_1[min][0]
                    v = Index_array_1[min][1]
                    index_2 = 0
                    for t in range(u-1,u+2):
                        for s in range(v-1,v+2):
                            Frame_t_1 = np.array(imgt_1[t-half_size:t+half_size,s-half_size:s+half_size],dtype = 'int16')
                            Index_array_2[index_2] = [t,s]
                            if Frame_t_1.shape == (block_size, block_size):
                                SAD_2[index_2] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                sum += 1
                            else:
                                SAD_2[index_2] = float('inf')
                            index_2 +=1
                    min2 = np.argmin(SAD_2)

                    Motion_Vector = np.append(Motion_Vector,(Index_array_2[min2][0]-i_o,Index_array_2[min2][1]-j_o))


                else:
                    u = Index_array_1[min][0]
                    v = Index_array_1[min][1]

                    while search_range > 0.5:
                        index_2 = 0
                        for t in range(u-search_range,u+search_range+1,search_range):
                            for s in range(v-search_range,v+search_range+1,search_range):
                                Frame_t_1 = np.array(imgt_1[t-half_size:t+half_size,s-half_size:s+half_size],dtype = 'int16')
                                Index_array_2[index_2] = [t,s]
                                if Frame_t_1.shape == (block_size, block_size):
                                    SAD_2[index_2] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                    sum += 1
                                else:
                                    SAD_2[index_2] = float('inf')
                                index_2 +=1
                        min2 = np.argmin(SAD_2)
                        u = Index_array_2[min2][0]
                        v = Index_array_2[min2][1]
                        search_range = int(search_range/2)

                    Motion_Vector = np.append(Motion_Vector,(u-i_o,v-j_o))

    Motion_Vector = Motion_Vector.reshape(p, q, 2)
    print(sum)
    return Motion_Vector

def EBMA_Reconstruction(imgt_1, Motion_vector,block_size):

    imgt_1 = np.array(imgt_1,dtype= 'int16')
    y,x = imgt_1.shape
    Y = int(y/block_size)
    X = int(x/block_size)
    imgt = np.zeros_like(imgt_1)

    for i in range(0,Y):
        for j in range(0,X):
            MV_Y = Motion_vector[i][j][0]
            MV_X = Motion_vector[i][j][1]
            I = block_size * i
            J = block_size * j
            Re_block = imgt_1[I+MV_Y:I+MV_Y+block_size,J+MV_X:J+MV_X+block_size]
            if Re_block.shape == (block_size,block_size):
                imgt[I:I+block_size,J:J+block_size] = Re_block
            else:
                imgt[I:I + block_size, J:J + block_size] = imgt_1[I:I+block_size,J:J+block_size]
    imgt = np.array(imgt,dtype = np.uint8)
    return imgt

def Reconstruction(imgt_1, Motion_vector,block_size):

    imgt_1 = np.array(imgt_1,dtype= 'int16')
    y,x = imgt_1.shape
    Y = int(y/block_size)
    X = int(x/block_size)
    imgt = np.zeros_like(imgt_1)
    half_size = int(block_size/2)
    for i in range(half_size, y, block_size):
        for j in range(half_size, x, block_size):
            I = int(i/block_size)
            J = int(j/block_size)
            MV_Y = Motion_vector[I][J][0]
            MV_X = Motion_vector[I][J][1]
            Re_block = imgt_1[i+MV_Y-half_size:i+MV_Y+half_size,j+MV_X-half_size:j+MV_X+half_size]
            if Re_block.shape == (block_size, block_size):
                imgt[i-half_size:i+half_size, j-half_size:j+half_size] = Re_block
            else:
                imgt[i-half_size:i+half_size, j-half_size:j+half_size] = imgt_1[i-half_size:i+half_size, j-half_size:j+half_size]
        imgt = np.array(imgt, dtype=np.uint8)

    return imgt

def PSNR(Original,Reconstructed):

    Original = np.array(Original, dtype = 'float')
    Reconstructed = np.array(Reconstructed, dtype = 'float')
    y,x = Original.shape

    MSE = np.sum(np.square(np.absolute(Original - Reconstructed))) / (y*x)
    test = (np.square(np.max(Original) - np.min(Original)) / MSE)
    test1 = np.square(np.max(Original) - np.min(Original))
    test2 = np.max(Original) - np.min(Original)
    PSNR = 10 * np.log10((np.square(np.max(Original)- np.min(Original))/MSE))

    print(MSE)
    print(PSNR)

    return

def vector_quiver(data):
    y,x,z = data.shape
    X = np.arange(x)
    Y = y-1-np.arange(y)
    U = np.empty((y,x))
    V = np.empty((y,x))
    for i in range(y):
       for j in range(x):
           U[i][j] = data[i][j][1]
           V[i][j] = -data[i][j][0]
    plt.figure()
    plt.quiver(X,Y,U,V,angles='xy')
    plt.show()

    return

def image_split(img):

    img = np.array(img, dtype = np.uint8)
    y,x = img.shape
    y= int(y/16)
    x= int(x/16)
    split = np.hsplit(img,x)
    print(len(split))
    print(split[0])
    for i in range(len(split)):
        vsplit =  np.vsplit(split[i],y)
    return

def EBMA_Fractional(imgt,imgt_1,block_size,search_range,step_size):
###### imgt = anchor frame, imgt_1 = target frame #####

## step_size =1 : integer_pel accuracy search
    y,x = imgt.shape
    print(imgt.shape,imgt_1.shape)
    imgt = np.array(imgt, dtype = np.uint8)
    imgt_1 = np.array(imgt_1,dtype = np.uint8)
    Search_range= 2*search_range + 1
    SAD = np.zeros((Search_range,Search_range))
    Motion_vector = []
    Motion_vector = np.array(Motion_vector, dtype='int8')
    for i in range(0,y,block_size):
        for j in range(0,x,block_size):
            Frame_t = np.array(imgt[i:i+block_size,j:j+block_size],dtype ='int16')
            p = int(y/block_size)
            q = int(x/block_size)
            for z in range(i-search_range,i+search_range+1,step_size):
                for v in range(j-search_range,j+search_range+1,step_size):
                    Frame_t_1 = np.array(imgt_1[z:z+block_size,v:v+block_size],dtype = 'int16')
                    #Sub = Frame_t - Frame_t_1
                    #print(i,j,z,v)
                    if Frame_t_1.shape != (block_size,block_size):
                        SAD[z-i+search_range][v-j+search_range] = float('inf')
                    else:
                        SAD[z-i+search_range][v-j+search_range] = np.sum(np.absolute(Frame_t - Frame_t_1))

            a = np.argwhere(SAD==np.min(SAD))-search_range
            Motion_vector =np.append(Motion_vector, (a[0][0],a[0][1]))

    Motion_vector = Motion_vector.reshape(p,q,2)

    return Motion_vector

def video():

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        out.write(frame)

        cv2.imshow('frame',frame)
        cv2.imshow('gray',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllwindows()


    return

def speed_test(c):
    test_array_flatten = np.arange(c)
    z = test_array_flatten.shape[0]

    test_array = np.arange(c).reshape((int(c**(1/2)),int(c**(1/2))))
    y, x = test_array.shape
    test_array_3 = np.arange(c).reshape(int(np.ceil(c**(1/3))),int(np.ceil(c**(1/3))),int(np.ceil(c**(1/3))))

    t,y,u = test_array_3.shape

    start = timeit.default_timer()
    for q in range(t):
       for w in range(y):
           for e in range(u):
                test_array_3[q][w][e]=0

    #stop = timeit.default_timer()
    #print(stop - start)

    start = timeit.default_timer()
    for p in range(z):
        test_array_flatten[p] = 0

    stop = timeit.default_timer()
    print(stop - start)

    start = timeit.default_timer()
    for i in range(y):
        for j in range(x):
            test_array[i][j] = 0

    stop = timeit.default_timer()
    print(stop - start)
    return


def test():

    test_array = np.array([(1,2),(2,4),(4,4),(3,5)])
    print(test_array.shape)
    print(np.any(test_array == (3,3)))
    a= np.where(test_array == [5,5])
    print(test_array[0][0],test_array[0][1])
    test_array = np.array([1,2,3])
    print(np.where(test_array == 1))
    print(a)
    return

main()
