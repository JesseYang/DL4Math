require 'torch'
require 'nn'
require 'nnx'
require 'my_sequencer'
require 'optim'
require 'rnn'
require 'model'
require 'data'

-- model_0()
model_1()
s = nn.Sequencer(m)
c = nn.CTCCriterion()

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
	pred = torch.Tensor(1, table.getn(inputTable), klass)
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
	learningRate = 1e-6,
	learningRateDecay = 0,
	weightDecay = 0,
	momentum = 0
}

iteration = 1
function train(num)
	for i = 1,num do
		x, f = optim.sgd(feval, x, config)
		print("loss for iteration " .. iteration .. " is: " .. f[1])
		iteration = iteration + 1
	end
end
