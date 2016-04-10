require 'nn'
require 'rnn'

local MySequencer, parent = torch.class("nn.MySequencer", "nn.Container")

function MySequencer:__init(model_idx)
	self.model_idx = model_idx
	self.klass = 10

	self.module = self:buildModel(model_idx)
	self.modules = { self.module }

	self.sharedClones = { }
end

function MySequencer:getStep(step)
	local m = self.sharedClones[step]
	if not m then
		m = self.module:stepClone()
		self.sharedClones[step] = m
	end
	return m
end

function MySequencer:buildModel(model_idx)
	if (model_idx == 1) then
		self.window = 30
		self.stride = 2
		return nn.Linear(2,4)
	end
end

function MySequencer:updateOutput(input)
	size = input:size()
	self.height = size[2]
	self.width = size[3]
	output_width = torch.floor(((self.width - self.window) + 1) / self.stride)
	self.output = torch.Tensor(1, output_width, self.klass)
	for i = 1, output_width do
		curInput = input:sub(1, 1, 1, self.height, 1+(i-1)*self.stride, self.window+(i-1)*stride)
		self.output[1][i] = self.getStep(i):updateOutput(curInput)
	end
	return self.output
end

function MySequencer:updateGradInput(input, gradOutput)
	self.gradInput = input:clone():zero()
	self.gradInput[1][1] = self:getStep(1):updateGradInput(inputTable[1][1], gradOutput[1][1])
	self.gradInput[1][2] = self:getStep(2):updateGradInput(inputTable[1][2], gradOutput[1][2])

	return self.gradInput
end

function MySequencer:accGradParameters(inputTable, gradOutput, scale)
	self:getStep(1):accGradParameters(inputTable[1][1], gradOutput[1][1],  scale)
	self:getStep(2):accGradParameters(inputTable[1][2], gradOutput[1][2],  scale)

	return self.gradInput
end
