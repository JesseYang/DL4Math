local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'torch'
require 'util'
require 'image'
require 'gnuplot'
require 'lfs'
require 'model'

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
	local type_idx = 1

	local segment_count = 0
	for dirname in lfs.dir(type_str .. "_set") do
		if (dirname ~= "." and dirname ~= "..") then
			for filename in lfs.dir(type_str .. "_set/" .. dirname) do
				if (filename ~= "." and filename ~= "..") then
					segment_count = segment_count + 1
				end
			end
		end
	end
	segment_count = segment_count / 2
	print("Loading " .. type_str .. " data set (" .. segment_count .. " images)")


	for dirname in lfs.dir(type_str .. "_set") do
		if (dirname ~= "." and dirname ~= "..") then
		-- if (dirname ~= "." and dirname ~= ".." and dirname == "0060_04") then
			for filename in lfs.dir(type_str .. "_set/" .. dirname) do
				if (filename ~= "." and filename ~= "..") then
					-- if (string.find(filename, "2_6.jpg") ~= nil or string.find(filename, "3_1.jpg") ~= nil) then
					if (string.find(filename, ".jpg") ~= nil) then
						print(filename)
						-- read the image into the byte tensor "img"
						local raw_img = cv.imread{type_str .. "_set/" .. dirname .. "/" .. filename, cv.IMREAD_GRAYSCALE}
						local size = raw_img:size()
						local height = size[1]
						local width = size[2]
						local img = torch.ByteTensor(1, height, width)
						cv.threshold{raw_img, img[1], 10, 255, cv.THRESH_BINARY}
						ori_imgs_type[type_idx] = img:clone()
						img = img:float()
						-- cv.imshow{"img", img[1]}
						-- cv.waitKey {0}
						-- padding the img to the height padding_height
						local padding_img = torch.Tensor(1, padding_height, width + 2 * horizon_pad):fill(255)
						padding_img
							:narrow(3, horizon_pad + 1, width)
							:narrow(2, math.max(0, math.ceil((padding_height - height) / 2)), height)
							:copy(img)
						-- cv.imshow{"img", padding_img[1]}
						-- cv.waitKey {0}
						local mean = padding_img:sum() / (width * height)
						local padding_img = (padding_img - mean) / 100
						imgs_type[type_idx] = padding_img
						-- read the label into "labels_type"
						local label_pathname = type_str .. "_set/" .. dirname .. "/" .. mysplit(filename, ".")[1] .. ".txt"
						local label_file = assert(io.open(label_pathname, "r"))
						local label = label_file:read()
						-- print(label)
						labels_type[type_idx] = get_label_by_str(label)
						label_pathname_ary_type[type_idx] = label_pathname
						type_idx = type_idx + 1
					end
				end
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
	label_pathname_ary_train = { }

	ori_imgs_type = ori_imgs_train
	imgs_type = imgs_train
	labels_type = labels_train
	type_idx_ary = train_idx_ary
	label_pathname_ary_type = label_pathname_ary_train
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

function getInputTableFromImg(img)
	local size = img:size()
	local height = size[2]
	local width = size[3]

	local inputTable = { }
	for i = 1, width - window + 1, stride do
		inputTable[1 + (i - 1) / stride] = use_cuda and 
			img:sub(1, 1, 1, height, i, i + window - 1):cuda() or
			img:sub(1, 1, 1, height, i, i + window - 1)
	end
	return inputTable
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

	train_idx = (train_idx == table.getn(imgs_train)) and 1 or (train_idx + 1)

	local inputTable = getInputTableFromImg(img)
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
