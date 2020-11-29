load net;


inputs = zeros(size(testX,1), size(testX, 2), size(testX, 3), size(testX, 4)* 9);
targets = zeros(size(testY,2),10);
labels = zeros(size(testY,2),10);
cur = 0;
m_lb = eye(10);
for i = 1:size(testY,2)
    predLabelsTest = net.classify(testX(:,:,:,i));
    if (predLabelsTest ~= categorical(testY(i))) 
        continue;
    end
    for j = 1:10
        if (testY(i)+1 == j)
            continue;
        end
        cur = cur + 1;
        inputs(:,:,:,cur) = testX(:,:,:,i);
        targets(cur, :) = m_lb(j,:);
        labels(cur, :) = m_lb(testY(i)+1,:);
    end
end

inputs = inputs(:,:,:,1:cur);
targets = targets(1:cur,:);
labels = labels(1:cur,:);

save targeted_data;

