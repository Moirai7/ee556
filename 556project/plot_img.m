

list = [14,30,36,31,20,9,22,1,62,10];
newton = [];
adam = [];
ori = [];
for id = 1:length(list)
    img = concat(id-1, list(id),'newton');
    newton = [newton; img];
    img = concat(id-1, list(id),'adam');
    adam = [adam; img];
    if id ~= 1
        img = imread(['res_adjust/adam/img' int2str(list(id)) '_target_0_ori.png']);
    else
        img = imread(['res_adjust/adam/img' int2str(list(id)) '_target_1_ori.png']);
    end
    ori = [ori;img];
end

subplot('position',[0. 0.05 0.2 0.9]); 
%subplot(1,3,1);
imshow(ori);
subplot('position',[0.2 0.05 0.3 0.9]); 
%subplot(1,3,2);
imshow(adam);
subplot('position',[0.6 0.05 0.3 0.9]); 
%subplot(1,3,3);
imshow(newton);

p = get(gcf,'Position')
set(gcf, 'Position', [p(1),(2),1560,500]);

function imgs = concat(i, idx, pt)
    imgs = [];
    for j = 0:9
        if i == j
            if i ~= 0
                img = imread(['res_adjust/' pt '/img' int2str(idx) '_target_0_ori.png']);
            else
                img = imread(['res_adjust/' pt '/img' int2str(idx) '_target_1_ori.png']);
            end
        else
            img = imread(['res_adjust/' pt '/img' int2str(idx) '_target_' int2str(j) '_adv.png']);
        end
        imgs = [imgs, img];
    end
end