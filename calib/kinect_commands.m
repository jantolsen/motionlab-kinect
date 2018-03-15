% Commands for Kinect V2

% Info
info = imaqhwinfo('kinect')

% Color video
vid_color = videoinput('kinect',1);

% Preview video
preview(vid_color)

% Depth video
vid_depth = videoinput('kinect', 1);
preview(vid_depth)

% Take snapshot
img = getsnapshot(vid);
imwrite(img,'test.PNG')