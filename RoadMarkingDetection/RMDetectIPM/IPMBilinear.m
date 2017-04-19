function [ipmImg] = IPMBilinear(inputImg)
%     close all;
    figure, imshow(inputImg), 
    title('Click on 4 corners clockwise, starting from top left corner');
%     hold on
%     line([350,450],[300,300],'Color','r','LineWidth',2)
%     line([450,800],[300,420],'Color','g','LineWidth',2)
%     line([800,1],[420,420],'Color','b','LineWidth',2)
%     line([1,350],[420,300],'Color','y','LineWidth',2)
    [imgX, imgY] = ginput;
    roadCorners = [imgX imgY];
%     roadCorners = [300, 300;
%                     500, 300;
%                     800, 420;
%                     1, 420]; 

    %Generate image of letter size paper
    planeCorners = zeros(4,2);
    planeCorners(1,:) = [1,1];
    planeCorners(2,:) = [800, 1];
    planeCorners(3,:) = [800, 600];
    planeCorners(4,:) = [1, 600];
    
    %Estimate Homography
    H = getHomography(roadCorners, planeCorners)
    
%     H = [-0.2172, -0.6927, 287.8360;
%         0.0108, -1.2216, 371.7445;
%         0.0000, -0.0035, 1.0000];

%    H = [-0.2704   -1.0024  377.3301;
%    -0.0237   -1.6243  500.0155;
%    -0.0001   -0.0036    1.0000];

% H = [-0.1169   -0.6977  245.1427;
%       0.0000   -1.4085  423.9043;
%       0.0000   -0.0035    1.0000];
    
    %ipmImg = zeros(size(inputImg,1)-350, size(inputImg,2)-200);
    ipmImg = zeros(size(inputImg,1)*2, size(inputImg,2)*2);
    
    % Iterate through rows and cols of the ipmImg
    %for rows=100:size(inputImg,1)-250
        %for cols=1:size(inputImg,2)-200
    for rows=1:size(ipmImg,1)
         for cols=1:size(ipmImg,2)
            points = H\[cols, rows, 1]'; % inv(H)
            pointsNorm = points ./points(3);
            
            % Bilinear Interpolation
            if(pointsNorm(1)>=1 && pointsNorm(2)>=1 ... 
                    && pointsNorm(1)<=800 && pointsNorm(2)<=600)
                
                x = pointsNorm(1);
                y = pointsNorm(2);
                % Find coordinates of 4 neighboring pixels
                x1 = floor(pointsNorm(1));
                x2 = ceil(pointsNorm(1));
                y1 = floor(pointsNorm(2));
                y2 = ceil(pointsNorm(2));
                % Form 4 neighbouring pixels
                P1 = [x1,y1];
                P2 = [x2,y1];
                P3 = [x2,y2];
                P4 = [x1,y2];
                
                % Perform Bilinear Interpolation
                % Formula Source: https://en.wikipedia.org/wiki/Bilinear_interpolation
                if(x1 ~= x2 || y1 ~= y2)
                    pixelValR = double(1/((x2-x1)*(y2-y1))) * ...
                               double([x2 - x, x-x1]) * ...
                               double([inputImg(P1(2),P1(1),1), inputImg(P2(2),P2(1),1); inputImg(P3(2),P3(1),1), inputImg(P4(2),P4(1),1)]) * ...
                               double([y2-y;y-y1]);
                    
                    pixelValG = double(1/((x2-x1)*(y2-y1))) * ...
                               double([x2 - x, x-x1]) * ...
                               double([inputImg(P1(2),P1(1),2), inputImg(P2(2),P2(1),2); inputImg(P3(2),P3(1),2), inputImg(P4(2),P4(1),2)]) * ...
                               double([y2-y;y-y1]);
                           
                    pixelValB = double(1/((x2-x1)*(y2-y1))) * ...
                               double([x2 - x, x-x1]) * ...
                               double([inputImg(P1(2),P1(1),3), inputImg(P2(2),P2(1),3); inputImg(P3(2),P3(1),3), inputImg(P4(2),P4(1),3)]) * ...
                               double([y2-y;y-y1]);
        
                    ipmImg(rows,cols,1) = pixelValR;
                    ipmImg(rows,cols,2) = pixelValG;
                    ipmImg(rows,cols,3) = pixelValB;
                else
                    % If pixel coodinates obtained by inverse
                    % transformation equation are integers then they map to
                    % original image pixels correctly, no need to do
                    % bilinear interpolation for this case.
                    ipmImg(rows,cols,:) = inputImg(pointsNorm(2), pointsNorm(1),:);
                end
            end
        end
    end
    
    figure, imshow(uint8(ipmImg))
end