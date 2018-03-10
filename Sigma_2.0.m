img=imread('../data/butterfly.jpg'); %Scanning the image
GrayImg=im2double(rgb2gray(img)); %Converting to grayscale
%imshow(GrayImg);
sigma=3;
numScales=10;
[h w]=size(GrayImg);

%scaleMultiplier=sqrt(sqrt(2));
scaleMultiplier=1.5;
threshold=0.01;   


%[h w]= size(GrayImg);
scaleSpaceI=zeros(h,w,numScales);
scaleSpaceII=scaleSpaceI;
Kernelsize=2*ceil(3*sigma)+1;
LoGKernel=fspecial('log',Kernelsize,sigma); %implementing the Laplacian of Guassian filter
LoGKernel=sigma.^2 * LoGKernel;
choice= input('Enter your choice\n 1: Downsizing the image \n 2: Increasing Filter Size \n'); 
switch choice
case 1
tic;    
for i=1:numScales
	if i==1
		downsizedImg=GrayImg;
	else
		downsizedImg=imresize(GrayImg,1/(scaleMultiplier^(i-1)),'bicubic');
	end

	filteredImg=imfilter(downsizedImg,LoGKernel,'same','replicate'); %Used replicate to make sure there are no blobs forming at the edges of the image
	filteredImg=filteredImg .^ 2;
	upscaledImg=imresize(filteredImg,[h w],'bicubic'); %For faithful reproduction of a smaller image
	%imshow(upscaledImg);
	scaleSpaceI(:,:,i)=upscaledImg;
end
scaleSpace=scaleSpaceI;
%[h w]=size(GrayImg);
%scaleSpaceII=zeros(h,w,numScales);

case 2
tic;
downsizedImg=GrayImg;
%[h w]=size(GrayImg);
%scaleSpaceII=zeros(h,w,n);

for i=1:numScales
	scaledSigma=sigma*scaleMultiplier^(i-1); 
	Kernelsize=2*ceil(3*sigma)+1;
	LoGKernel=fspecial('log',Kernelsize,scaledSigma);
	LoGKernel=scaledSigma.^2 * LoGKernel;
	%Increasing the filter size

	filteredImg=imfilter(downsizedImg,LoGKernel,'same','replicate');
	filteredImg=filteredImg .^ 2;
	scaleSpaceII(:,:,i)=filteredImg;
	%imshow(filteredImg);	
	%LoGKernel=scaledSigma^2
end
scaleSpace=scaleSpaceII;
end

%scaleSpace=scaleSpaceI;

nms=zeros(h,w,numScales);
domain=ones(5,5); %used 5 to eliminate my problem of too many overlapping circles
neighbourSize=5;
for i=1:numScales
	
nms(:,:,i)=ordfilt2(scaleSpace(:,:,i),neighbourSize^2,domain);

end
for i=1:numScales
	nms(:,:,i)=max(nms(:,:,max(i-1,1):min(i+1,numScales)),[],3);
end
nms = nms .* (nms==scaleSpace);

circx= [];
circy= [];
rad= [];

for i=1:numScales
	[rows,cols]=find(nms(:,:,i)>threshold); %Finding the row and column vectors of the centers of the circles 
	numberofBlobs=length(rows);
	radii=sqrt(2)*sigma*scaleMultiplier^(i-1);
	radii=repmat(radii,numberofBlobs,1);
	circx=[circx;rows]; 
	circy=[circy;cols];
	rad=[rad;radii];
end
    
show_all_circles(img,circy,circx,rad,'r',1.5);toc;
    
