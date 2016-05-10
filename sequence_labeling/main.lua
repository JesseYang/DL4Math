use_cuda = true
require 'torch'
require 'nn'
if (use_cuda) then
	require 'cutorch'
	require 'cunn'
end
require 'nnx'
require 'optim'
require 'rnn'
require 'model'
require 'data'
require 'image'


model_4()
c = use_cuda == true and nn.CTCCriterion():cuda() or nn.CTCCriterion()

-- Prepare the data
load_training_data()
load_test_data()

function copy_table(t)
	local res = { }
	for i, v in ipairs(t) do
		res[i] = v
	end
	return res
end

function equal_table(t1, t2)
	if (table.getn(t1) ~= table.getn(t2)) then
		return false
	end
	for i = 1, table.getn(t1) do
		if (t1[i] ~= t2[i]) then
			return false
		end
	end
	return true
end

function softmax(pred)
	local prob = pred:clone()
	local size = pred:size()
	local T = size[2]
	local L = size[3] - 1
	for t = 1, T do
		local temp = { }
		local sum = 0
		for k = 1, L + 1 do
			temp[k] = math.exp(pred[1][t][k])
			sum = sum + temp[k]
		end
		for k = 1, L + 1 do
			prob[1][t][k] = temp[k] / sum
		end
	end
	return prob
end

function prefix_search_decode(pred_param)
	local pred = pred_param:clone()

	local size = pred:size()
	local T = size[2]
	local L = size[3] - 1

	local gamma_p_star_b = { }
	local gamma_p_star_n = { }
	local p_star = { }
	local i_star = { }
	local i_star_prob = 1
	for t = 1, T do
		i_star_prob = i_star_prob * pred[1][t][1]
		gamma_p_star_n[t] = 0
		gamma_p_star_b[t] = i_star_prob
	end
	local p_star_prefix_prob = 1 - i_star_prob
	-- local P = { }
	-- P[p_star] = i_star_prob

	while (p_star_prefix_prob > i_star_prob) do
		local prob_remaining = p_star_prefix_prob
		local p_prob_store = { }
		local p_prefix_prob_store = { }
		local gamma_p_n_store = { }
		local gamma_p_b_store = { }
		local P = { }
		for k = 1, L do
			-- append k to p_star to get p
			local p = copy_table(p_star)
			p[table.getn(p) + 1] = k
			local gamma_p_b = { }
			local gamma_p_n = { }
			if (p_star == { }) then
				gamma_p_n[1] = pred[1][1][k + 1]	-- y_k^1
			else
				gamma_p_n[1] = 0
			end
			gamma_p_b[1] = 0
			local prefix_prob = gamma_p_n[1]
			for t = 2, T do
				local new_label_prob
				if (p_star[table.getn(p_star)] == k) then
					new_label_prob = gamma_p_star_b[t-1]
				else
					new_label_prob = gamma_p_star_b[t-1] + gamma_p_star_n[t-1]
				end
				gamma_p_n[t] = pred[1][t][k + 1] * (new_label_prob + gamma_p_n[t - 1])
				gamma_p_b[t] = pred[1][t][1] * (gamma_p_b[t - 1] + gamma_p_n[t - 1])
				prefix_prob = prefix_prob + pred[1][t][k + 1] * new_label_prob
			end
			local p_prob = gamma_p_n[T] + gamma_p_b[T]
			local p_prefix_prob = prefix_prob - p_prob
			p_prob_store[p] = p_prob
			p_prefix_prob_store[p] = p_prefix_prob
			gamma_p_n_store[p] = gamma_p_n
			gamma_p_b_store[p] = gamma_p_b

			prob_remaining = prob_remaining - p_prefix_prob
			if (p_prob > i_star_prob) then
				-- i^* = p
				i_star = p
				i_star_prob = p_prob
			end
			if (p_prefix_prob > i_star_prob) then
				P[p] = p_prefix_prob
			end
			if (prob_remaining <= i_star_prob) then
				break
			end
		end
		--[[
		for p in pairs(P) do
			if (equal_table(p, p_star)) then
				P[p] = nil
			end
		end
		]]
		local p_prefix_prob = -1
		for p in pairs(P) do
			if (P[p] > p_prefix_prob) then
				p_prefix_prob = P[p]
				p_star = p
				-- the p_star_prob, p_star_prefix_prob, gamma_p_star_n, and gamma_p_star_b also needs to be set
				p_star_prob = p_prob_store[p]
				p_star_prefix_prob = p_prefix_prob_store[p]
				gamma_p_star_n = gamma_p_n_store[p]
				gamma_p_star_b = gamma_p_b_store[p]
			end
		end
		if (p_prefix_prob == -1) then
			break
		end
	end

	-- i_star is the result, convert to predicted string and return
	local pred_str_ary = { }
	for i = 1, table.getn(i_star) do
		pred_str_ary[i] = label_set[i_star[i]]
	end
	return table.concat(pred_str_ary)
end

function showDataResult(img_idx)

	local ori_img = ori_imgs_type[img_idx]
	local img = imgs_type[img_idx]
	local label = labels_type[img_idx]

	local inputTable = getInputTableFromImg(img)
	local input_size = getInputSize(inputTable)


	local outputTable = s:forward(inputTable)
	pred = use_cuda and torch.CudaTensor(1, input_size, klass) or torch.Tensor(1, input_size, klass)
	for i = 1, input_size do
		pred[1][i] = torch.reshape(outputTable[i], 1, klass)
	end
	-- image.display(ori_img)
	label_str = ""
	for i = 1, table.getn(label) do
		label_str = label_str .. label_set[label[i]]
	end
	print("Label File Name: " .. label_pathname_ary_type[img_idx])
	print("Correct Label: " .. label_str)
	print("Decoded Label: " .. prefix_search_decode(softmax(pred)))

	local rank_num = 3
	pred_data = { }
	pred_str_1 = ""
	local pred_clone = pred:clone()
	for r = 1,rank_num do
		local pred_str = ""
		for i = 1, input_size do
			local temp, idx = torch.max(pred_clone[1][i], 1)
			pred_clone[1][i][idx[1]] = -1e10
			if (idx[1] == 1) then
				pred_str = pred_str .. " "
				if (r == 1) then
					pred_data[i] = -1
				end
			else
				pred_str = pred_str .. label_set[idx[1] - 1]
				if (r == 1) then
					pred_data[i] = idx[1] - 1
				end
			end
		end
		print(pred_str)
		if (r == 1) then
			pred_str_1 = pred_str
		end
	end
	local idx_str = ""
	for i = 1, input_size do
		idx_str = idx_str .. (i % 10)
	end
	print(idx_str)

	-- put the prediction results and the original image into one image
	--[[
	img_size = ori_img:size()
	height = img_size[2]
	width = img_size[3]
	label_size = table.getn(label_set)
	com_img = torch.Tensor(1, padding_height + (label_size + 1) * 3, width + 2 * horizon_pad):fill(255)
	com_img
		:narrow(3, horizon_pad + 1, width)
		:narrow(2, math.max(0, math.ceil((padding_height - height) / 2)) + (label_size + 1) * 3, height)
		:copy(ori_img)

	local data_idx = 1
	for i = 1, width + 2 * horizon_pad - window + 1, stride do
		if (pred_data[data_idx] ~= -1) then
			com_img[1][(pred_data[data_idx] - 1) * 3 + 1][i + math.floor(window / 2)] = 0
			com_img[1][(pred_data[data_idx] - 1) * 3 + 2][i + math.floor(window / 2)] = 0
			com_img[1][(pred_data[data_idx] - 1) * 3 + 3][i + math.floor(window / 2)] = 0
		end
		data_idx = data_idx + 1
	end

	image.display(com_img)

	for i = 1,table.getn(inputTable) do
		print(i .. ": " .. string.sub(pred_str_1, i, i))
		image.display(inputTable[i])
		local ii = io.read()
		if (ii == "q") then
			break
		end
	end
	]]
end

function showTestResult(img_idx)
	ori_imgs_type = ori_imgs_test
	imgs_type = imgs_test
	labels_type = labels_test
	label_pathname_ary_type = label_pathname_ary_test
	return showDataResult(img_idx)
end

function showTestResultByName(img_name)
	local img_idx = -1
	for i=1,table.getn(prefix_ary_test) do
		if (prefix_ary_test[i] == img_name) then
			img_idx = i
			break
		end
	end
	if (img_idx ~= -1) then
		showTestResult(img_idx)
	else
		print("Image does not exist")
	end
end

function showTrainResult(img_idx)
	ori_imgs_type = ori_imgs_train
	imgs_type = imgs_train
	labels_type = labels_train
	label_pathname_ary_type = label_pathname_ary_train
	return showDataResult(img_idx)
end

function showTrainResultByName(img_name)
	local img_idx = -1
	for i=1,table.getn(prefix_ary_train) do
		if (prefix_ary_train[i] == img_name) then
			img_idx = i
			break
		end
	end
	if (img_idx ~= -1) then
		showTrainResult(img_idx)
	else
		print("Image does not exist")
	end
end

function calDataErrRate()
	print("Error rate on " .. type_str .. " set (image number: " .. table.getn(imgs_type) .. ")")
	local err_num = 0
	for img_idx = 1,table.getn(imgs_type) do
		local img = imgs_type[img_idx]
		local label = labels_type[img_idx]

		local inputTable = getInputTableFromImg(img)
		local outputTable = s:forward(inputTable)
		local input_size = getInputSize(inputTable)
		local pred = use_cuda and torch.CudaTensor(1, input_size, klass) or torch.Tensor(1, input_size, klass)
		for i = 1, input_size do
			pred[1][i] = torch.reshape(outputTable[i], 1, klass)
		end

		label_str = ""
		for i = 1, table.getn(label) do
			label_str = label_str .. label_set[label[i]]
		end

		local pred_str_ary = { }
		local pred_idx = 1
		local last_c = ""
		for i = 1, input_size do
			local temp, idx = torch.max(pred[1][i], 1)
			pred[1][i][idx[1]] = -1e10
			if (idx[1] ~= 1) then
				if (last_c ~= label_set[idx[1] - 1]) then
					pred_str_ary[pred_idx] = label_set[idx[1] - 1]
					pred_idx = pred_idx + 1
					last_c = label_set[idx[1] - 1]
				end
			else
				last_c = ""
			end
		end
		local pred_str = table.concat(pred_str_ary)
		if (pred_str ~= label_str) then
			print("PRED: " .. pred_str)
			print("LABEL: " .. label_str)
			print(img_idx .. ": " .. label_pathname_ary_type[img_idx])
			err_num = err_num + 1
		end
	end
	print("Error rate: " .. err_num / table.getn(imgs_type) .. ". " .. err_num .. "/" .. table.getn(imgs_type))
end

function calTestErrRate()
	imgs_type = imgs_test
	labels_type = labels_test
	label_pathname_ary_type = label_pathname_ary_test
	type_str = "test"
	return calDataErrRate()
end

function calTrainErrRate()
	imgs_type = imgs_train
	labels_type = labels_train
	label_pathname_ary_type = label_pathname_ary_train
	type_str = "training"
	return calDataErrRate()
end

x, dl_dx = s:getParameters()

feval = function(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end
	-- inputTable, target = toySample()
	inputTable, target = nextSample()

	dl_dx:zero()
	-- forward of model
	outputTable = s:forward(inputTable)
	-- change the format of output of the nn.Sequencer to match the format of input of CTCCriterion
	local input_size = getInputSize(inputTable)
	pred = use_cuda and torch.CudaTensor(1, input_size, klass) or torch.Tensor(1, input_size, klass)
	for i = 1, input_size do
		pred[1][i] = torch.reshape(outputTable[i], 1, klass)
	end
	-- forward and backward of criterion
	loss_x = c:forward(pred, target)
	gradCTC = c:backward(pred, target)
	-- change the format of gradInput of the CTCCriterion to match the format of output of nn.Sequencer
	gradOutputTable = { }
	for i = 1, input_size do
		if (use_rnn) then
			gradOutputTable[i] = torch.reshape(gradCTC[1][i], 1, klass)
		else
			gradOutputTable[i] = torch.reshape(gradCTC[1][i], klass)
		end
	end
	s:backward(inputTable, gradOutputTable)
	return loss_x, dl_dx
end

-- sgd parameters
sgd_params = {
	learningRate = 1e-3,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0.9
}

-- adadelta parameters
adadelta_params = {
	rho = 0.95,
	eps = 1e-6
}
state = { }

loss_ary = { }
test_err_rate = { }
train_err_rate = { }
loss_epoch = { }

evalCounter = 0
function train(time)
	last_epoch = epoch or 0
	star_num = star_num or 0
	line_star = 80
	local huge_error = false
	for i =1,time do
		evalCounter = evalCounter + 1
		-- _, fs = optim.sgd(feval, x, sgd_params)
		_, fs = optim.adadelta(feval, x, adadelta_params, state)
		if (i % 1 == 0) then
			if (last_epoch ~= epoch) then
				star_num = 0
				io.write("Epoch " .. epoch .. ": ")
				last_epoch = epoch
			end
			local percent = math.floor((evalCounter % table.getn(imgs_train)) / table.getn(imgs_train) * 100)
			if (evalCounter % table.getn(imgs_train) == 0) then
				percent = 100
			end
			local cur_star_num = math.floor(percent / (100 / line_star))
			for j = star_num + 1,cur_star_num do
				io.write("=")
			end
			star_num = cur_star_num
			io.flush()
		end
		loss_ary[evalCounter] = fs[1]
		if (loss_ary[evalCounter] > 1000000000) then
			huge_error = true
			print("HUGE ERROR! 111")
			break
		end
	end
	return huge_error
end

function train_epoch(epoch_num)
	for e = 1, epoch_num do
		nClock = os.clock() 
		local huge_error = train(table.getn(imgs_train))
		if (huge_error == true) then
			print("HUGE ERROR! 222")
			break
		end
		local elapse = torch.round((os.clock() - nClock) * 10) / 10

		for j = star_num + 1, line_star do
			io.write("=")
		end
		io.flush()
		loss_tensor = torch.Tensor(loss_ary)
		num = table.getn(imgs_train)
		local loss_cur_epoch = loss_tensor:sub((last_epoch - 1) * num + 1, last_epoch * num):mean()
		io.write(". Ave loss: " .. loss_cur_epoch .. ".")
		loss_epoch[epoch] = loss_cur_epoch
		io.write(" Execution time: " .. elapse .. "s.")
		io.write("\n")

		-- save the model file
		torch.save("models/" .. epoch .. ".mdl", m)
	end
end

function load_model(model_idx)
	m = torch.load("models/" .. model_idx .. ".mdl")
	x, dl_dx = m:getParameters()
	s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
end

-- m = torch.load("models/debug.mdl")
-- s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)

-- load_model(13)
-- train_epoch(2)
-- torch.save("models/debug.mdl", m)

-- load_model(7)
train_epoch(500)
