
classdef ZOO
    properties
        %parameters
        batch_size = 128;
        max_iterations = 3000;
        early_stop_iters = 100;
        abort_early = true;
        print_every = 1000000;
        confidence = 0;
        lr = 1e-2;
        initial_zoo_const = 1.;
        binary_search_steps = 8;
        repeat = false;
        targeted = 1;
        reset_adam_after_found = false;
        use_importance = true;
        solver = "newton";%'adam';%newton
        start_iter = 0;
        init_size = 32;
        %use_importance = false; 
        img_sz = 28;
        sz;
        use_tanh = false;
        use_log = true;
        
        %valid variables
        modifier_up;
        modifier_down;
        real_modifier;
        mt;
        vt;
        adam_epoch;
        var_list;
        net;
        sample_prob;
        img;
        tlab;
        
        %random permutation for coordinate update
        %perm;
        %perm_idx = 0;
        
        %ADAM
        beta1 = 0.9;
        beta2 = 0.999;
        stage = 0;
        
        %placeholder
        zoo_const ;
        
        adjusted;
        e;
        types;
    end
    methods
        function obj = ZOO(net,adjusted)
            obj.sz = 28 * 28
            %obj.perm = perms(1:obj.sz)
            obj.mt = zeros(obj.sz,1);
            obj.vt = zeros(obj.sz,1);
            obj.adam_epoch = ones(obj.sz,1);
            obj.zoo_const = obj.initial_zoo_const;
            obj.var_list = 1: obj.sz;
            obj.net = net;
            obj.sample_prob = ones(obj.sz, 1)/obj.sz;
            obj.adjusted = adjusted;
            if obj.adjusted
                obj.e = 0.0001;
                obj.types = 'double';
            else
                obj.e = 1;
                obj.types = 'uint8';
            end
        end
        
        function b = argmax(~, x)
            [~,b] = max(x);
        end
        
        function [obj, o_bestattack, o_best_zoo_const, queries] = attack(obj, img, tlab)
            obj.zoo_const = obj.initial_zoo_const;
            lower_bound = 0.;
            upper_bound = 1e10;
            if obj.use_tanh
                img = atanh(img*1.99999999);
            end
            
            if ~obj.use_tanh
                obj.modifier_up = 0.5-img;
                obj.modifier_down= -0.5-img;
            end
            
            obj.img = img;
            obj.tlab = tlab;
                
            obj.real_modifier = zeros(obj.img_sz, obj.img_sz,obj.types);    
            
            o_best_zoo_const = obj.zoo_const;
            o_bestl2 = 1e10;
            o_bestscore = -1;
            o_bestattack = img;
            queries = 0;
            
            for outer_step = 1:obj.binary_search_steps
                disp(o_bestl2);
                bestl2 = 1e10;
                bestscore = -1;
                
                if obj.repeat == true && outer_step == obj.binary_search_steps-1
                    obj.zoo_const = upper_bound;
                end
                
                prev = 1e6;
                last_loss1 = 1.0;
                
                %reset
                obj.real_modifier = zeros(obj.img_sz, obj.img_sz,obj.types);    
                obj.mt = zeros(obj.sz,1,obj.types);
                obj.vt = zeros(obj.sz,1,obj.types);
                obj.adam_epoch = ones(obj.sz,1);
                obj.stage = 0;
                eval_costs = 0;
                for iteration = obj.start_iter: obj.max_iterations
                    if mod(iteration,obj.print_every) == 0
                        [obj, losses, l2s, loss1, loss2, scores, nimgs, real, other]=obj.get_loss(obj.real_modifier);
                        disp(scores);
                        disp('iter');
                        disp(iteration);
                        disp('loss');
                        disp(losses);
                        disp('loss1');
                        disp(loss1);
                        disp('loss2');
                        disp(loss2);
                        disp(real);
                        disp(other);
                        %imshow(cast(round((nimgs+0.5)*255),'uint8'));
                        disp('=============');
                    end
                    
                    [obj, l, l2, loss1, loss2, score, nimg] = obj.blackbox_optimizer(obj.real_modifier);
                    eval_costs = eval_costs+obj.batch_size;
                    if loss1 == 0.0 && last_loss1 ~= 0.0 && obj.stage == 0
                        if obj.reset_adam_after_found
                            obj.mt = zeros(obj.sz,1,obj.types);
                            obj.vt = zeros(obj.sz,1,obj.types);
                            obj.adam_epoch = ones(obj.sz,1);
                        end
                        obj.stage = 1;
                    end
                    last_loss1 = loss1;
                    if l2 < bestl2 &&  obj.argmax(score) == obj.argmax(tlab)
                        bestl2 = l2;
                        bestscore = obj.argmax(score);
                    end
                    if l2 < o_bestl2 && obj.argmax(score) == obj.argmax(tlab)
                        o_bestl2 = l2;
                        o_bestscore = obj.argmax(score);
                        o_bestattack = nimg;
                        o_best_zoo_const = obj.zoo_const;
                    end
                    if obj.abort_early && mod(iteration , obj.early_stop_iters) == 0
                        if l > prev*0.999
                            disp("Early stopping because there is no improvement");
                            break
                        end
                        prev = l;
                    end
                end
                if bestscore == obj.argmax(tlab) && bestscore ~= -1
                    queries = queries+eval_costs;
                    if o_bestl2 < 20.0
                        break
                    end
                    disp('old constant:');
                    disp(obj.zoo_const);
                    upper_bound = min(upper_bound,obj.zoo_const);
                    if upper_bound < 1e9
                        obj.zoo_const = (lower_bound + upper_bound)/2;
                    end
                    disp('new constant:');
                    disp(obj.zoo_const);
                else
                    disp('old constant:');
                    disp(obj.zoo_const);
                    lower_bound = max(lower_bound,obj.zoo_const);
                    if upper_bound < 1e9
                        obj.zoo_const = (lower_bound + upper_bound)/2;
                    else
                        obj.zoo_const = obj.zoo_const*10;
                    end
                    disp('new constant:');
                    disp(obj.zoo_const);
                end
            end
        end
        
        function [obj, l, l2, loss1, loss2, score, nimg] = blackbox_optimizer(obj,real_modifier)
            var = repmat(real_modifier, 1, obj.batch_size * 2 + 1);
            if obj.use_importance
                [var_indice,~] = datasample(obj.var_list, obj.batch_size, 'Replace', false, 'Weights', obj.sample_prob);
            else
                var_indice = randsample(obj.var_list, obj.batch_size, false);
            end
            indice = obj.var_list(var_indice);
            for i = 1:obj.batch_size
                %var = reshape(var(i * 2 + 1),-1)
                var((i * 2-1)*obj.sz+ indice(i)) = var((i * 2-1)*obj.sz+ indice(i)) + obj.e;
                var((i * 2)*obj.sz+ indice(i)) = var((i * 2)*obj.sz+ indice(i)) - obj.e;
            end
            var = reshape(var,obj.img_sz, obj.img_sz, []);
            [obj, losses, l2s, loss1, loss2, scores, nimgs, real, other]=obj.get_loss(var);
            if obj.solver == "adam"
                obj = obj.ADAM(losses, indice);
            elseif obj.solver == "newton"
                obj = obj.Newton(losses, indice);
            end
            %if obj.use_importance
            %    
            %end
            l = losses(1);
            l2 = loss2(1);
            loss1 = loss1(1);
            loss2 = loss2(1);
            score = scores(1,:);
            nimg = nimgs(:,:,:,1);
        end
        
        function [obj,loss, l2dist, loss1, loss2, output, newimg, real, other] = get_loss(obj, scaled_modifier)
            if obj.use_tanh
                newimg = tanh(scaled_modifier + obj.img)/2;
            else
                newimg = scaled_modifier + obj.img;
            end
            newimg = reshape(newimg, obj.img_sz, obj.img_sz, 1, []);
            if obj.adjusted
                if size(newimg,4) > 1
                    output = [predict(obj.net, newimg(:,:,:,1)); predict(obj.net, newimg(:,:,:,2:end))];
                else
                    output = predict(obj.net, newimg);
                end
            else
                adjust_newimg = cast(newimg,obj.types);
                if size(newimg,4) > 1
                    output = [predict(obj.net, adjust_newimg(:,:,:,1)); predict(obj.net, adjust_newimg(:,:,:,2:end))];
                else
                    output = predict(obj.net, adjust_newimg);
                end
            end
            if obj.use_tanh
                l2dist = squeeze(sum((newimg - tanh(obj.img)/2).^2,[1 2 3]));
            else
                l2dist = squeeze(sum((newimg - obj.img).^2,[1 2 3]));
            end
            real = sum(obj.tlab.*output,2);
            other = max((1-obj.tlab).*output-obj.tlab*10000,[],2);
            if obj.use_log
                loss1 = max(0., log(other+1e-30)-log(real+1e-30));
            else
                loss1 = max(0., other-real+obj.confidence);
            end
            loss2 = l2dist;
            loss1 = squeeze(obj.zoo_const*loss1);
            loss = loss1+loss2;
        end
        
        function [obj]=ADAM(obj, losses, indice)
            grad = [];
            for i = 1: obj.batch_size
                grad(end+1) = (losses(i*2)-losses(i*2+1))/(obj.e*2);
            end
            mt_adam = obj.mt(indice)';
            mt_adam = obj.beta1 * mt_adam + (1-obj.beta1) * grad;
            obj.mt(indice) = mt_adam;
            vt_adam = obj.vt(indice)';
            vt_adam = obj.beta2 * vt_adam + (1-obj.beta2) * (grad.*grad);
            obj.vt(indice) = vt_adam;
            epoch = obj.adam_epoch(indice)';
            corr = (sqrt(1 - power(obj.beta2, epoch))) ./ (1 - power(obj.beta1, epoch));
            %obj.real_modifier = reshape(obj.real_modifier, obj.sz, 1);
            old_val = obj.real_modifier(indice); 
            old_val = old_val-obj.lr * corr .* mt_adam ./ (sqrt(vt_adam) + 1e-8);
            if  ~obj.use_tanh
                old_val = max(min(old_val, obj.modifier_up(indice)), obj.modifier_down(indice));
            end
            if obj.adjusted
                obj.real_modifier(indice) = old_val;
            else
                obj.real_modifier(indice) = cast(round(old_val),obj.types);
            end
            obj.adam_epoch(indice) = epoch + 1;
            %obj.real_modifier = reshape(obj.real_modifier, obj.img_sz, obj.img_sz);
        end
        
        function [obj]= Newton(obj, losses, indice)
            cur_loss = losses(1);
            grad = [];
            hess = [];
            for i = 1: obj.batch_size
                grad(end+1) = (losses(i*2)-losses(i*2+1))/(obj.e*2);
                hess(end+1) = (losses(i*2)-2*cur_loss+losses(i*2+1))/((obj.e*2)*(obj.e*2));
            end
            hess(hess <= 0) = 1.0;
            hess(hess < 0.1) = 0.1;
            old_val = obj.real_modifier(indice); 
            old_val = old_val-obj.lr * grad ./ hess;
            if ~obj.use_tanh
                old_val = max(min(old_val, obj.modifier_up(indice)), obj.modifier_down(indice));
            end
            obj.real_modifier(indice) = old_val;
        end
    end
end



