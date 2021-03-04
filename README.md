
# This repository is python code of block matching algorithm and Huffman Encoding

* Block Matching Algorithm
    
    * Motion vector
        ![Motion_vector](https://user-images.githubusercontent.com/62092317/109893250-aa778980-7cce-11eb-961d-b2a2412abb99.PNG)
        
        * Changing Block size
        ![Motion_vector_block_size](https://user-images.githubusercontent.com/62092317/109893253-ab102000-7cce-11eb-8811-54bad90b9b82.PNG)
        
        * Changing Search range
        ![Motion_vector_Search_range](https://user-images.githubusercontent.com/62092317/109893258-aba8b680-7cce-11eb-9b79-44141f18a642.PNG)

    * Exhaustive search (EBMA)

        ```python
        def EBMA(imgt,imgt_1,block_size,search_range,step_size):
            ###### imgt = anchor frame, imgt_1 = target frame #####
            ## step_size =1 : integer_pel accuracy search
            y,x = imgt.shape
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
        ```
    * Logarithmic search
      ![Logarithmic_search](https://user-images.githubusercontent.com/62092317/109893247-a9def300-7cce-11eb-8c75-e78240dca01c.PNG)
      ```python
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
      ```

   * Three Step Search and New Three Step Search

      ![TSS_and_NTSS](https://user-images.githubusercontent.com/62092317/109893272-afd4d400-7cce-11eb-87ea-021263bcb8c9.PNG)
      ```python
      def Three_Step_Search(imgt,imgt_1,block_size,search_range):
        y, x = imgt.shape
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
            for j in range(half_size, x, block_size):
                Frame_t = np.array(imgt[i - half_size:i + half_size, j - half_size:j + half_size], dtype='int16')
                i_o = i
                j_o = j
                index = 0
                search_range = 4
                print(i,j)
                for u in range(i - 1, i + 2):
                    for v in range(j - 1, j + 2):
                            Frame_t_1 = np.array(imgt_1[u - half_size:u + half_size, v - half_size:v + half_size],dtype='int16')
                            Index_array_1[index] = [u,v]
                            if Frame_t_1.shape != (block_size, block_size):
                                SAD_1[index] = float('inf')
                            else:
                                SAD_1[index] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                sum += 1
                            index += 1

                for u in range(i-4,i+5,4):
                    for v in range(j-4,j+5,4):
                        if (u,v)!= (i,j):
                            Frame_t_1 = np.array(imgt_1[u - half_size: u+half_size, v-half_size:v+half_size],dtype = 'int16')
                            Index_array_1[index] = [u, v]
                            if Frame_t_1.shape != (block_size, block_size):
                                SAD_1[index] = float('inf')
                            else:
                                SAD_1[index] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                sum += 1
                            index += 1

                min = np.argmin(SAD_1)

                if min == 4 :
                    Motion_Vector = np.append(Motion_Vector,(0,0))

                else:
                    if min <= 8 :
                        u = Index_array_1[min][0]
                        v = Index_array_1[min][1]
                        index_2 = 0
                        for t in range(u-1,u+2):
                            for s in range(v-1,v+2):
                                Frame_t_1 = np.array(imgt_1[t-half_size:t+half_size,s-half_size:s+half_size],dtype = 'int16')
                                Index_array_2[index_2] = [t,s]
                                if  Frame_t_1.shape != (block_size, block_size):
                                    SAD_2[index_2] = float('inf')
                                else:
                                    SAD_2[index_2] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                    sum += 1
                                index_2 +=1
                        min2 = np.argmin(SAD_2)

                        Motion_Vector = np.append(Motion_Vector,(Index_array_2[min2][0]-i_o,Index_array_2[min2][1]-j_o))


                    elif min > 8 :
                        u = Index_array_1[min][0]
                        v = Index_array_1[min][1]

                        while search_range > 0.5:
                            index_2 = 0
                            for t in range(u-search_range,u+search_range+1,search_range):
                                for s in range(v-search_range,v+search_range+1,search_range):
                                    Frame_t_1 = np.array(imgt_1[t-half_size:t+half_size,s-half_size:s+half_size],dtype = 'int16')
                                    Index_array_2[index_2] = [t,s]
                                    if  Frame_t_1.shape != (block_size, block_size):
                                        SAD_2[index_2] = float('inf')
                                    else:
                                        SAD_2[index_2] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                        sum += 1
                                    index_2 +=1
                            min2 = np.argmin(SAD_2)
                            u = Index_array_2[min2][0]
                            v = Index_array_2[min2][1]
                            search_range = int(search_range/2)

                        Motion_Vector = np.append(Motion_Vector,(u-i_o,v-j_o))

        Motion_Vector = Motion_Vector.reshape(p, q, 2)
        print(sum)
        return Motion_Vector
      ```
    * Diamond Search
      ![Diamond_search](https://user-images.githubusercontent.com/62092317/109893245-a9465c80-7cce-11eb-9bb8-958fcda1bcdb.PNG)
      ```python
      def Diamond_Search (imgt,imgt_1,block_size):

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

                while min1 != 4 :
                    index_1 = 0

                    for u in range(i-2,i+3):
                        for v in range(j-2,j+3):
                            overlap= 100
                            if (u+v == i+j-2 or u+v == i+j or u+v == i+j+2) and (u,v) != (i-2,j+2) and (u,v) != (i+2,j-2):

                                for k in range(0,9):
                                    if Index_array_1[k][0] == u and Index_array_1[k][1] == v :
                                        overlap = k

                                if overlap == 100:
                                    Frame_t_1 = np.array(imgt_1[u-half_size:u+half_size,v-half_size:v+half_size],dtype ='int16')
                                    Index_array_1[index_1] = [u,v]
                                    if Frame_t_1.shape != (block_size, block_size):
                                        SAD_1[index_1] = float('inf')
                                        SAD_temp[index_1] = float('inf')
                                    else:
                                        #Sub = Frame_t - Frame_t_1
                                        SAD_1[index_1] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                        SAD_temp[index_1] = SAD_1[index_1]
                                        sum += 1
                                elif overlap !=100 :
                                    SAD_1[index_1] = SAD_temp[overlap]
                                    Index_array_1[index_1] = [u, v]

                                index_1 += 1

                    min1 = np.argmin(SAD_1)
                    i = Index_array_1[min1][0]
                    j = Index_array_1[min1][1]

                index_2 = 0
                for t in range(i-1,i+2):
                    for s in range(j-1,j+2):
                        if t+s == i+j-1 or t+s == i+j+1 or (t,s) == (i,j) :
                                Frame_t_1 = np.array(imgt_1[t-half_size:t+half_size,s-half_size:s+half_size],dtype = 'int16')
                                Index_array_2[index_2] = [t,s]
                                if Frame_t_1.shape != (block_size, block_size):
                                    SAD_2[index_2] = float('inf')
                                else:
                                    SAD_2[index_2] = np.sum(np.absolute(Frame_t - Frame_t_1))
                                    sum += 1
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
      ```
    * Simple and Efficient Search
      ![Simple_and_Efficient_search](https://user-images.githubusercontent.com/62092317/109893267-aea3a700-7cce-11eb-8328-9e328d075f83.PNG)
      ```python
      def Simple_and_Efficient_search (imgt,imgt_1,block_size):

        y, x = imgt.shape
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
        return Motion_Vector
      ```
    
* Reconstruction

    * Reconstruction ( Changing Block size)
    ![Reconstruction_Block_size](https://user-images.githubusercontent.com/62092317/109893262-ac414d00-7cce-11eb-94dd-46032a0a9b86.PNG)
    
    * Reconstruction ( Changing Search range)
    ![Reconstruction_search_range](https://user-images.githubusercontent.com/62092317/109893265-ad727a00-7cce-11eb-9e67-e6d71d316b1f.PNG)

    ```python
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
    ```

* EBMA Reconstruction
    ```python
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
    ```

* Computation comparison
    ![Computation_comparison](https://user-images.githubusercontent.com/62092317/109893242-a8152f80-7cce-11eb-863c-a2b4c2d87d98.PNG)

#See details in [HERE](https://github.com/SeongSuKim95/Digital_Image_processing_Block_matching_algorithm_and_Reconstruction/blob/master/Various_Block_matching_Algorithm.pdf)
