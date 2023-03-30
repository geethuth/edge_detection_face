% The following code uploads the photo, display the photo and finds the
% edges of face from SVM segmented image using various edge detection
% methods as well as the custom method created by Geethu

%%%%%%%%%%%%%%% start of main function %%%%%%%%%%%%%%%%%%

function edge_face_svm_geethu()

close all; % close all windows
clear all;

my_photo = imread("geethu.jpeg"); % loading photo
my_photo = imresize(my_photo,0.25); % making the process faster
[M, N, dim] = size(my_photo)
im =  rgb2hsv(my_photo);% converting to HSV colour space

% Ploting the sub plots
figure(1),
subplot(5,4,[1 5]), imshow(my_photo), title('Original'); % display original image
subplot(5,4,[13 17]), imshow(add_sign(my_photo)), title('Signed Image'); % display signature added image by calling add_sign() function
subplot(5,4,2), imshow(im), title('HSV'); % display HSV converted image

% Converting an image into a 2D table with each row has RGB values
hs=[reshape(im(:,:,1),1,[]);reshape(im(:,:,2),1,[]); reshape(im(:,:,3),1,[])];
[dim, no] = size(hs)
X = hs';
X(1:10,:)

Y = zeros(no,1);
for i=1:no
    if X(i, 1)<0.2 %threshold
        Y(i)=1; % white pixel has ground truth 1 otherwise -1
    else
        Y(i)=-1;
    end
end

Y(1:10)
sz = [ size(Y)];

% Trains a SVM using the RGB values and the ground truth vector
svm =fitcsvm(X,Y,'Standardize',true,'KernelFunction','rbf','KernelScale','auto'); % implementing SVM
sz = size(svm);

% Evaluates the SVM model using k-fold cross-validation
cv = crossval(svm)
loss = kfoldLoss(cv) % calculates the loss.
[~, score] = kfoldPredict(cv); % predict each pixel score (1,-1)
sz_score = size(score);
mean(score<0)
predX = X;
predX = (score(:,2)>=0);
predX = predX .* X;

% reshape back to image resolution
im_pred(:,:,1) = reshape(predX(:,1),M,N);
im_pred(:,:,2) = reshape(predX(:,2),M,N);
im_pred(:,:,3) = reshape(predX(:,3),M,N);
im_pred_rgb = hsv2rgb(im_pred);
subplot(5,4,3), imshow(mat2gray(im_pred)), title('Predicted HSV'); % plotting predicted HSV
subplot(5,4,6), imshow(mat2gray(im_pred_rgb)), title('Predicted RGB'); % plotting predicted RGB

% display edges %
im_eg = rgb2gray(im_pred_rgb);
im_canny = edge(im_eg,'canny');

% creating a customized edge
my_edge = custom_edge(im_eg); % function for creating custom edge

my_edge_rgb = customedge_color(my_edge); % function for adding color to custom edge

% sub-plot different edge detection methods
subplot(5,4,7), imshow(edge(im_eg,'canny')); title('Edge= Canny'); % displaying Canny edge
subplot(5,4,10), imshow(im_canny,'ColorMap', [1 1 1; 0 0 1]); title('Edge= Colored Canny'); % displaying coloured canny edge
subplot(5,4,11), imshow(edge(im_eg,'sobel')); title('Edge= Sobel'); % displaying Sobel edge
subplot(5,4,14), imshow(edge(im_eg,'roberts')); title('Edge= Roberts'); % displaying Roberts edge
subplot(5,4,15), imshow(edge(im_eg,'log')); title('Edge= LOG'); % displaying Log edge
subplot(5,4,18), imshow(edge(im_eg,'prewitt')); title('Edge= Prewitt'); % displaying Prewitt edge
subplot(5,4,19), imshow(edge(im_eg,'zerocross')); title('Edge= Zero-Cross'); % displaying Zero cross edge
subplot(5,4,[4 8]), imshow(my_edge); title('Edge= custom'); % displaying edge created using my custom method
subplot(5,4,[16 20]), imshow(my_edge_rgb); title('Edge= colored custom'); % displaying colored custom edge

end
%%%%%%%%%%%%%%% end of main function %%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%% function for adding signature 'TH' to image %%%%%%%%%%%%%%
function sign_image = add_sign(f_image)
%% Creating first initial 'T'
im_T=zeros(50,50);
for i=201:250
    im_T(350:354,202:i-2)=255;
end
im_T(354:400,222:226)=255;

%% Creating second initial 'H'
im_H=zeros(50,50);
im_H(352:400,204:208)=255;
for i=201:250
    im_H(373:377,204:i-4)=255;
end
im_H(352:400,242:246)=255;

face_image=f_image;

f_image(1:400,1:248,1)=im_T(:,:);
f_image(1:400,1:248,2)=im_T(:,:);
f_image(1:400,1:248,3)=im_T(:,:);

f_image(1:400,249:494,1)=im_H(:,:);
f_image(1:400,249:494,2)=im_H(:,:);
f_image(1:400,249:494,3)=im_H(:,:);

%% Only need letters 'T' and 'H'
f_image=face_image;

for i=350:400
    for j=201:246
        if(im_T(i,j)>190) % thrsholding
            f_image(i,j,1)=im_T(i,j);   %% add to RED chanel
        end
        if(im_H(i,j)>190) % thrsholding
            f_image(i,j+50,2)=im_H(i,j);   %% add to GREEN chanel
        end
    end
end
sign_image = f_image;
end
%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% Function to create my own egde by customising Canny edge method %%%%%%%%%
function eg = custom_edge(im_eg) % segmented image as parameter to the function
% customizing Canny edge detection to improve edges
% [low_threshold high_threshold] parameter specifies the thresholds for hysteresis thresholding
low_threshold = .1;
high_threshold = .6;
sigma = 15; % setting the standard deviation of the filter
eg = edge(im_eg, 'canny', [low_threshold high_threshold], sigma);
% Remove small edge fragments
eg = bwareaopen(eg, 20);
% apply Gaussian smoothing to edges
eg = imgaussfilt(double(eg), 1, 'FilterSize', 5);
end
%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%

%%%%%%%% function for adding color to custom edge %%%%%%%%%
function my_edge_rgb = customedge_color(my_edge) % custom edge image as parameter to the function
% Get the height and width of the image
[height, width, ~] = size(my_edge);
% Create a matrix of zeros with the same dimensions as the image
zeros_img = zeros(height, width);
% Specify the color for the edges
color = [1 0 0]; % red color
% Overlay the binary edge map using the specified color
my_edge_rgb = imoverlay(zeros_img, my_edge, color);
end
%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%