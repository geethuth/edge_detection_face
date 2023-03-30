% The following code uploads the photo, display the photo and finds the
% edges of face from K-means segmented image using various edge detection
% methods as well as the custom method created by Geethu

function edge_face_kmeans_geethu()

Irgb = imread("geethu.jpeg"); %loading photo

figure(1),
subplot(2,5,[1 6]), imshow(Irgb), title('original');

I = rgb2hsv(Irgb);
%I = Irgb;
% segment the image into k=2,3,5,10 regions
figure(1),
subplot(2,5,2), imshow(I), title('HSV');

I = uint8(I); % converting uint8 for the function of imsegkmeans


[L,Centers] = imsegkmeans(I,2); % 2 regions
B = labeloverlay(I,L);
subplot(2,5,3), imshow(B), title('K=2');

% display edges
im_eg = rgb2gray(B);
subplot(2,5,4), imshow(edge(im_eg,'sobel')); title('Edge= Sobel');
im22 = edge(im_eg,'canny');
[row, col] = size(im22);
%im22(row-20:row,:) = 0;
% Apply Canny edge detection with built-in functions
eg = edge(im_eg, 'canny', [.1 .6], 15);
eg = bwareaopen(eg, 20);
 % Smooth the edges with a Gaussian filter
eg = imgaussfilt(double(eg), 1, 'FilterSize', 5);

subplot(2,5,7), imshow(im22); title('Edge= Canny');  % displaying Canny edge
subplot(2,5,8), imshow(edge(im_eg,'roberts')); title('Edge= Roberts');  % displaying roberts edge
subplot(2,5,9), imshow(edge(im_eg,'log')); title('Edge= LOG'); % displaying log edge
subplot(2,5,[5 10]), imshow(eg); title('Edge= my method'); % displaying custom edge


