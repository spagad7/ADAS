close all
img = imread('RoadMarkings/roadmark_0101.jpg');
figure, imshow(img)

A = [0.197907,  -0.178787,      360.82;
        0,      -0.0050404166,  50.9823;
        0,      -0.000446185,      1   ];


img2 = uint8(zeros(size(img)));
    
for y=1:size(img, 1)
    for x=1:size(img, 2)
        coords(:,:,:) = A * [y, x, 1]';
        if(coords(1)>0 && coords(2)>0)
            img2(round((1+coords(2))), round((1+coords(1))), :) = img(y,x,:);
            img2(round((1+coords(2))), round((1+coords(1))), :) = img(y,x,:);
        end
    end
end

figure, imshow(uint8(img2))