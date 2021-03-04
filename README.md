# DIP_BMA,Reconstruction,HuffmanEncoding
2020 MMI Lab. DIP 세미나 
BMA(Block Matching Algorithm) and Reconstruction by Block Matching

1.발표 내용 – 주어진 자료와 https://scholar.google.com에서 논문을 검색

각 구현 내용을 함수나 class 별로 나눠서 함수의 재사용이 가능하도록 구현, 구현 내용은 다음과 같습니다.

1. Block Matching Algorithm

동영상에서 연속하는 두 프레임 사이의 motion vector를 측정합니다. 이 때 block size와 search range, step size를 변화시킬 수 있도록 구현합니다. 추정한 motion vector를 quiver함수를 이용하여 확인합니다.

2.Reconstruction by Block Matching

BMA로 구한 motion vector를 사용하여 동영상을 압축하고 반대로 복원합니다.

3.공통사항

주어진 그림을 사용하여 올바른 결과가 나오는지 확인합니다. 또한 영상에 대해 연산량을 측정하고 최대한 효율적으로 구현합니다. 의미 있는 결과가 있다면 발표자료에 첨부합니다. 교재만으로 이해가 부족한 경우 별도의 검증된 자료를 이용하시길 바랍니다.

4.Huffman Encoding

Huffman Encdoing을 통해 이미지를 압축하고 압축률과 각 intensity별로 encode 된 code를 txt로  
