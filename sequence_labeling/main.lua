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


-- model_0()
model_1()
s = use_cuda == true and nn.Sequencer(m):cuda() or nn.Sequencer(m)
c = use_cuda == true and nn.CTCCriterion():cuda() or nn.CTCCriterion()
-- c = nn.CTCCriterion()

-- Prepare the data
load_training_data()

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
	local input_size = table.getn(inputTable)
	pred = use_cuda and torch.CudaTensor(1, input_size, klass) or torch.Tensor(1, input_size, klass)
	for i = 1, table.getn(inputTable) do
		pred[1][i] = torch.reshape(outputTable[i], 1, klass)
	end
	-- forward and backward of criterion
	loss_x = c:forward(pred, target)
	gradCTC = c:backward(pred, target)
	-- change the format of gradInput of the CTCCriterion to match the format of output of nn.Sequencer
	gradOutputTable = { }
	for i = 1, table.getn(inputTable) do
		gradOutputTable[i] = torch.reshape(gradCTC[1][i], klass)
	end
	s:backward(inputTable, gradOutputTable)
	return loss_x, dl_dx
end

-- sgd parameters
sgd_params = {
	learningRate = 1e-4,
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
	end
end

function train_epoch(epoch_num)
	for e = 1, epoch_num do
		nClock = os.clock() 
		train(table.getn(imgs_train))
		local elapse = torch.round((os.clock() - nClock) * 10) / 10

		for j = star_num + 1, line_star do
			io.write("=")
		end
		io.flush()
		local loss_tensor = torch.Tensor(loss_ary)
		local num = table.getn(imgs_train)
		local loss_cur_epoch = loss_tensor:sub((last_epoch - 1) * num + 1, last_epoch * num):mean()
		io.write(". Ave loss: " .. loss_cur_epoch .. ".")
		loss_epoch[epoch] = loss_cur_epoch
		io.write(" Execution time: " .. elapse .. "s.")
		io.write("\n")

		-- save the model file
		torch.save("models/" .. epoch .. ".mdl", m)
	end
end
