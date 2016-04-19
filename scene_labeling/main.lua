require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'gnuplot'
require 'lfs'
require 'data'
require 'util'
require 'model'

model_3()

-- Criterion
criterion = nn.ClassNLLCriterion()

-- Prepare the data
load_training_data()
-- load_test_data()

function plotTrainResult(img_idx, show_pixel_err)
	ori_imgs_type = ori_imgs_train
	label_imgs_type = label_imgs_train
	imgs_type = imgs_train
	plotResult(img_idx, show_pixel_err)
end

function plotTestResult(img_idx, show_pixel_err)
	ori_imgs_type = ori_imgs_test
	label_imgs_type = label_imgs_test
	imgs_type = imgs_test
	plotResult(img_idx, show_pixel_err)
end

function plotResult(img_idx, show_pixel_err)
	show_pixel_err = show_pixel_err or false
	if (colors == nil) then
		load_color_config()
	end
	local img = imgs_type[img_idx]
	local ori_img = ori_imgs_type[img_idx]
	local s = img:size()
	local height = s[2]
	local width = s[3]
	local result = torch.zeros(3, height, width)
	local ground_truth = torch.zeros(3, height, width)
	local skip = 0
	local cur_skip = 0
	image.display(img)
	for x = 1, width do
		for y = 1, height do
			if (ori_img[1][y][x] ~= 255) then
				local i = img:sub(1, 1, y - (length-1)/2, y + (length-1)/2, x - (length-1)/2, x + (length-1)/2)
				score = m:forward(i)
				local m_t, m_i = torch.max(score, 1)
				for c = 1,3 do
					result[c][y][x] = colors[m_i[1]][c]
					ground_truth[c][y][x] = colors[label_imgs_type[img_idx][y][x]][c]
				end
				if (show_pixel_err) then
					cur_skip = cur_skip + 1
					if (m_i[1] ~= label_imgs_type[img_idx][y][x] and cur_skip > skip) then
						local err_i = torch.Tensor(3, length, length):fill(255)
						for xx = x - (length-1)/2, x + (length-1)/2 do
							for yy = y - (length-1)/2, y + (length-1)/2 do
								err_i[1][yy - (y - (length-1)/2) + 1][xx - (x - (length-1)/2) + 1] = ori_img[1][yy][xx]
								err_i[2][yy - (y - (length-1)/2) + 1][xx - (x - (length-1)/2) + 1] = ori_img[1][yy][xx]
								err_i[3][yy - (y - (length-1)/2) + 1][xx - (x - (length-1)/2) + 1] = ori_img[1][yy][xx]
							end
						end
						err_i[1][(length+1)/2][(length+1)/2] = 255
						image.display(err_i, 3)
						print("******************** Error Pixel. Location: (" .. y .. ", " .. x .. ") ********************")
						for k = 1,klass do
							if (label_imgs_type[img_idx][y][x] == k) then
								print('*\t' .. label_name_ary[k] .. "\t" .. score[k])
							elseif (m_i[1] == k) then
								print('!\t' .. label_name_ary[k] .. "\t" .. score[k])
							else
								print('\t' .. label_name_ary[k] .. "\t" .. score[k])
							end
						end
						local ii = io.read()
						if (ii ~= "") then
							local k = tonumber(ii)
							if (k == nil) then
								show_pixel_err = false
							else
								skip = k
								cur_skip = 0
							end
						end
					end
				end
			end
		end
	end
	image.display(result, 3)
	image.display(ground_truth, 3)
end

x, dl_dx = m:getParameters()

feval = function(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end
	-- get next sample in the training set
	input, target = nextSample()
	-- reset gradients
	dl_dx:zero()
	-- evaluate the loss function and its derivative wrt x
	local pred = m:forward(input)
	local loss_x = criterion:forward(pred, target)
	m:backward(input, criterion:backward(pred, target))
	return loss_x, dl_dx
end

-- sgd parameters
sgd_params = {
	learningRate = 1e-4,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0.9
}

loss_ary = { }
test_err_rate = { }
train_err_rate = { }
loss_epoch = { }

function plotLossEpoch()
	gnuplot.plot(torch.Tensor(loss_epoch))
end

-- cycle on data
evalCounter = 0
function train(time)
	last_epoch = epoch or 0
	star_num = star_num or 0
	line_star = 80
	for i =1,time do
		evalCounter = evalCounter + 1
		_, fs = optim.sgd(feval, x, sgd_params)
		if (i % 100 == 0) then
			if (last_epoch ~= epoch) then
				star_num = 0
				io.write("Epoch " .. epoch .. ": ")
				last_epoch = epoch
			end
			local percent = math.floor((evalCounter % table.getn(train_data)) / table.getn(train_data) * 100)
			local cur_star_num = math.floor(percent / (100 / line_star))
			for j = star_num+1,cur_star_num do
				io.write("=")
			end
			star_num = cur_star_num
			io.flush()
		end
		loss_ary[evalCounter] = fs[1]
	end
end

function calTestErrRate()
	type_data = test_data
	imgs_type = imgs_test
	type_str = "test"
	return calDataErrRate()
end

function calTrainErrRate()
	type_data = train_data
	imgs_type = imgs_train
	type_str = "train"
	return calDataErrRate()
end

-- calculate the pixel error rate on test set
function calDataErrRate()
	local errNum = 0
	print("Error rate on " .. type_str .. " set (data size: " .. table.getn(type_data) .. ", image number: " .. table.getn(imgs_type) .. ")")
	local img_err_pixels = 0
	local cur_img_idx = 1
	for i = 1,table.getn(type_data) do
		-- for each pixel in test data
		local type_ele = type_data[i]
		local target = type_ele[4]
		local img_idx = type_ele[1]
		local stepX = type_ele[3]
		local stepY = type_ele[2]

		if (img_idx ~= cur_img_idx) then
			print("Image " .. cur_img_idx .. ": " .. img_err_pixels .. " wrong pixels.")
			cur_img_idx = img_idx
			img_err_pixels = 0
		end

		local input = imgs_type[img_idx]:sub(1, 1, stepY - (length-1)/2, stepY + (length-1)/2, stepX - (length-1)/2, stepX + (length-1)/2)

		-- pedict
		local pred = m:forward(input)
		local m_t, m_i = torch.max(pred, 1)

		if (m_i[1] ~= target) then
			errNum = errNum + 1
			img_err_pixels = img_err_pixels + 1
		end
	end
	print("Error rate: " .. errNum / table.getn(type_data) .. ". Wrong pixel number: " .. errNum)
	return errNum / table.getn(type_data)
end

function train_epoch(epoch_num)
	for e = 1, epoch_num do
		nClock = os.clock() 
		train(table.getn(train_data))
        local elapse = torch.round((os.clock() - nClock) * 10) / 10

		for j = star_num + 1, line_star do
			io.write("=")
		end
		io.flush()
		local loss_tensor = torch.Tensor(loss_ary)
		local num = table.getn(train_data)
		local loss_cur_epoch = loss_tensor:sub((last_epoch - 1) * num + 1, last_epoch * num):mean()
		io.write(". Ave loss: " .. loss_cur_epoch .. ".")
		loss_epoch[epoch] = loss_cur_epoch
		io.write(" Execution time: " .. elapse .. "s.")
		io.write("\n")

		-- test on the test set
		-- test_err_rate[epoch] = calTestErrRate()

		-- save the model file
		torch.save("results/" .. epoch .. ".mdl", m)
	end
end

function load_model(model_idx)
	m = torch.load("results/" .. model_idx .. ".mdl")
	x, dl_dx = m:getParameters()
	epoch = model_idx
	train_idx = 1
	evalCounter = epoch * table.getn(train_data)
	for i = 1,epoch do
		loss_epoch[i] = 0
	end
	for i = 1,epoch * table.getn(train_data) do
		loss_ary[i] = 0
	end
end
