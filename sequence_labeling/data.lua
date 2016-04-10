require 'torch'
require 'util'
require 'image'
require 'gnuplot'
require 'lfs'

train_mean = 0
stride = 2

function get_label_by_str(label_str)
	char_idx = 1
	local result = { }
	for c in label_str:gmatch(".") do
		for i = 1,table.getn(label_set) do
			if (label_set[i] == c) then
				result[char_idx] = i
			end
		end
		char_idx = char_idx + 1
	end
	return result
end

function load_data()
	local imgs_type_idx = 1
	local type_idx = 1

	local imgs_count = 0
	for label_filename in lfs.dir(type_str .. "_set/specs") do
		if (label_filename ~= "." and label_filename ~= "..") then
			imgs_count = imgs_count + 1
		end
	end
	print("Loading " .. type_str .. " data set (" .. imgs_count .. " images)")

	for label_filename in lfs.dir(type_str .. "_set/specs") do
		if (label_filename ~= "." and label_filename ~= "..") then
			local uuid = mysplit(label_filename, ".")[1]
			local image_filename = uuid .. "_gray.dat"
			local spec_filename = uuid .. ".txt"

			local image_file = assert(io.open(type_str .. "_set/images/" .. image_filename, "r"))
			local spec_file = assert(io.open(type_str .. "_set/specs/" .. spec_filename, "r"))

			local spec = spec_file:read()
			local spec_ary = mysplit(spec, ";")
			local label_str = spec_ary[2]
			local size_ary = mysplit(spec_ary[1], ",")
			local width = tonumber(size_ary[1])
			local height = tonumber(size_ary[2])
			local img = torch.zeros(1, height, width + 2 * pad):fill(255)
			local ori_img = torch.zeros(1, height, width + 2 * pad):fill(255)
			for y = 1, height do
				for x = 1 + pad, width + pad do
					img[1][y][x] = tonumber(string.byte(image_file:read(1)))
					ori_img[1][y][x] = img[1][y][x]
				end
			end
			local mean
			if (type_str == "training") then
				mean = img:sum() / (width * height)
				train_mean = mean
			else
				mean = train_mean
			end
			local img = (img - mean) / 100
			imgs_type[imgs_type_idx] = img
			ori_imgs_type[imgs_type_idx] = ori_img
			labels_type[imgs_type_idx] = get_label_by_str(label_str)

			imgs_type_idx = imgs_type_idx + 1
			if (imgs_type_idx % 10 == 0) then
				print(imgs_type_idx .. " images loaded")
			end
		end
	end

	for i = 1,table.getn(imgs_type) do
		type_idx_ary[i] = i
	end

	print("Finish loading " .. type_str .. " data set.")
end

function load_training_data()
	ori_imgs_train = { }
	imgs_train = { }
	labels_train = { }
	train_idx_ary = { }

	ori_imgs_type = ori_imgs_train
	imgs_type = imgs_train
	labels_type = labels_train
	type_idx_ary = train_idx_ary
	type_str = "training"

	load_data()

	train_idx = 1
end

inputTable = { }
inputTable[1] = torch.rand(1, 3, 3)
inputTable[2] = torch.rand(1, 3, 3)
local label = {{ 1, 3 }}
function toySample()
	return inputTable, label
end

function nextSample()
	epoch = epoch or 0

	-- whether goto next epoch
	if (train_idx == 1) then
		epoch = epoch + 1
		shuffle(train_idx_ary)
	end

	local img = imgs_train[train_idx_ary[train_idx]]
	local label = { labels_train[train_idx_ary[train_idx]] }
	local size = img:size()
	local height = size[2]
	local width = size[3]

	train_idx = (train_idx == table.getn(imgs_train)) and 1 or (train_idx + 1)

	local inputTable = { }
	for i = 1, width - window + 1, stride do
		inputTable[1 + (i - 1) / stride] = img:sub(1, 1, 1, height, i, i + window - 1)
	end
	-- return inputTable, label
	local temp = { }
	for i = 1, 50 do
		temp[i] = inputTable[i]
	end
	return inputTable, label
end

function get_label_from_pred(pred)
	local pred_label = { }
	local size = pred:size()
	local length = size[2]
	local klass = size[3]
	for i = 1, length do
		local max = 0
		for k = 1, klass do
			if (pred[1][i][k] > max) then
				pred_label[i] = k
				max = pred[1][i][k]
			end
		end
	end
	return pred_label
end
