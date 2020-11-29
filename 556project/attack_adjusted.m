load net_adjusted;

total_success = 0;
l2_total = 0.0;
zoo_attack = ZOO(net,true);
zoo_attack.solver = "newton";
%zoo_attack.solver = "adam";
m_lb = eye(10);
total = 0;
queries = 0;
for i = 200:size(testY,2)%
    input = double(adjust_testX(:,:,:,i));
    label = m_lb(testY(i)+1,:);
    output = predict(net, input);
    predLabelsTest = net.classify(input);
    disp(i);
    disp("true label:");
    disp(label);
    disp("probability:");
    disp(predLabelsTest);
    disp(output);
    if zoo_attack.argmax(output) ~= zoo_attack.argmax(label)
        disp('wrong prediction!')
        continue
    end
    for k = 1:10
        j = randsample(10,1);
        target = m_lb(j,:);
        if (zoo_attack.argmax(label) == zoo_attack.argmax(target))
            continue;
        end
        total = total +1;
        disp("true label:");
        disp(label);
        disp("probability:");
        disp(output);
        disp("target:");
        disp(target);
        
        [obj, adv, const, query] = zoo_attack.attack(input, target);
        
        l2_distortion = sqrt(sum(power((adv-input),2),'all'));
        newoutput = predict(net, adv);
        success = false;
        if (zoo_attack.argmax(newoutput) == zoo_attack.argmax(target))
            success = true;
        end
        if l2_distortion > 20.0
            success = false;
        end
        if success
            queries = queries + query;
            diffs = adv-input;
            imwrite(cast(round((diffs+0.5)*255),'uint8'), "res_adjust/"+zoo_attack.solver+"/img"+int2str(i)+"_target_"+int2str(j-1)+"_diff.png");
            save("res_adjust/"+zoo_attack.solver+"/img"+int2str(i)+"_target_"+int2str(j-1)+"_diff.mat",'diffs');
            imwrite(cast(round((adv+0.5)*255),'uint8'), "res_adjust/"+zoo_attack.solver+"/img"+int2str(i)+"_target_"+int2str(j-1)+"_adv.png");
            save("res_adjust/"+zoo_attack.solver+"/img"+int2str(i)+"_target_"+int2str(j-1)+"_adv.mat",'adv');
            imwrite(cast(round((input+0.5)*255),'uint8'), "res_adjust/"+zoo_attack.solver+"/img"+int2str(i)+"_target_"+int2str(j-1)+"_ori.png");
            save("res_adjust/"+zoo_attack.solver+"/img"+int2str(i)+"_target_"+int2str(j-1)+"_ori.mat",'input');
            total_success = total_success + 1;
            l2_total = l2_total + l2_distortion;
        end
        break
    end
end

disp(total_success);
disp(l2_total);
disp(total);

save attack_result_newton;