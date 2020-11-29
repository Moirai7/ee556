
path = 'res_adjust/newton/';

d = load('data.mat');
testY = d.testY;
testX = d.testX;
total = 0;
for i = 1:200
    if length(dir([path 'img' int2str(i) '_*_ori.png'])) ~= 0
        imwrite(testX(:,:,:,i),path+"img"+int2str(i)+"_target_"+int2str(testY(i))+"_adv.png");
        total = total + 1;
    end
end


idxs = [];
nums = [];
for i = 0:9
    imgs = dir([path,'*',int2str(i),'_adv.png']);
    idx = [];
    for j = 1: length(imgs)
        n=imgs(j).name;
        end_id = strfind(n,'_');
        idx(end+1) = str2num(['uint8(',n(4:end_id(1)-1),')']);
    end
    nums(end+1) = length(imgs);
    disp(length(imgs)/total)
end
