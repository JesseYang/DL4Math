--local cv = require 'cv'
local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'nn'
require 'lfs'
require 'util'
require 'image'

data_path = "data_set/"

function line_extraction_1(im_bw, line_dir_name)
	local size =  im_bw:size()
	local height = size[1]
	local width = size[2]
	local is_last_row_space = true
	local start_row = 0
	local line_idx = 1
	for cur_row = 1,height do
		if (torch.sum(im_bw[cur_row]) == 255 * width) then
			-- this should be a separation row, until last row should be a line
			if (is_last_row_space ~= true) then
				-- get new line, from start_row to last_row
				cv.imwrite { dpi .. "_binary/" .. line_dir_name .. "/line_" .. line_idx .. ".jpg", im_bw:sub(start_row, cur_row - 1, 1, width) }
				line_idx = line_idx + 1
			end
			is_last_row_space = true
		else
			if (is_last_row_space) then
				start_row = cur_row
			end
			is_last_row_space = false
		end
	end
	if (is_last_row_space ~= true) then
		cv.imwrite { dpi .. "_binary/" .. line_dir_name .. "/line_" .. line_idx .. ".jpg", im_bw:sub(start_row, height, 1, width) }
	end
end

function water_flow(im_bw)
	local im_bw_left = im_bw:clone():fill(0)
	local angle_level = 5
	for i = 1, width do
		if (i > 1) then
			im_bw_left:sub(1,height, i,i):copy(im_bw_left:sub(1,height, i-1,i-1))
		end
		--[[
		local sum
		if (i > angle_level) then
			sum = torch.sum(im_bw_left:sub(1, height, i-angle_level, i-1), 2)
		end
		]]
		for j = 1, height do
			if (im_bw[j][i] == 255) then
				if (i == 1) then
					-- background points in the first column are always wet 
					im_bw_left[j][i] = 122
				else
					if (im_bw_left[j][i] == 0) then
						if (i > angle_level) then
							-- check the above row
							if (j > 1) then
								if (torch.sum(im_bw_left:sub(j-1, j-1, i-angle_level, i-1)) == 122 * angle_level) then
								-- if (sum[j-1] == 122 * angle_level) then
									if (im_bw[j-1][i] == 255) then
										im_bw_left[j][i] = 122
									end
								end
							end
							-- check the below row
							if (im_bw_left[j][i] ~= 122 and j < height) then
								if (torch.sum(im_bw_left:sub(j+1, j+1, i-angle_level, i-1)) == 122 * angle_level) then
								-- if (sum[j+1] == 122 * angle_level) then
									if (im_bw[j+1][i] == 255) then
										im_bw_left[j][i] = 122
									end
								end
							end
						end
					end
				end
			else
				im_bw_left[j][i] = 0
			end
		end
	end
	return im_bw_left
end

function line_extraction_2(im_bw, im_bw_for_blur, filename)
	size = im_bw:size()
	height = size[1]
	width = size[2]
	-- first water flow from left to right
	local im_bw_left = water_flow(im_bw)
	local im_bw_right = image.hflip(water_flow(image.hflip(im_bw)))

	local im_bw_water = im_bw:clone()
	local im_for_label = im_bw:clone():fill(1)
	for i = 1, width do
		for j = 1, height do
			if (im_bw_left[j][i] == 122 and im_bw_right[j][i] == 122) then
				im_bw_water[j][i] = 122
				im_for_label[j][i] = 0
			end
		end
	end
	cv.imwrite { "water_imgs/" .. filename .. "_water.jpg", im_bw_water }
	return im_for_label
end

function load_label_img(label_filename)
	local pad = 10
	local image_filename = filename_prefix .. "_gray.dat"
	local spec_filename = filename_prefix .. ".txt"

	local label_file = assert(io.open("data_set/labels/" .. label_filename, "r"))
	local image_file = assert(io.open("data_set/images/" .. image_filename, "r"))
	local spec_file = assert(io.open("data_set/specs/" .. spec_filename, "r"))


	local spec = spec_file:read()
	local spec_ary = mysplit(spec, ",")
	local width = spec_ary[1] + pad * 2
	local height = spec_ary[2] + pad * 2
	ori_img = torch.ByteTensor(height, width):fill(255)
	equal_top_img = torch.ByteTensor(height, width):fill(255)
	equal_bottom_img = torch.ByteTensor(height, width):fill(255)
	fraction_img = torch.ByteTensor(height, width):fill(255)
	other_img = torch.ByteTensor(height, width):fill(255)
	for x = 1 + pad, width - pad do
		for y = 1 + pad, height - pad do
			local label = tonumber(string.byte(label_file:read(1)))
			ori_img[y][x] = tonumber(string.byte(image_file:read(1)))
			if (label == 1) then
				other_img[y][x] = 0
			end
			if (label == 2) then
				equal_top_img[y][x] = 0
			end
			if (label == 3) then
				equal_bottom_img[y][x] = 0
			end
			if (label == 4) then
				fraction_img[y][x] = 0
			end
			if (ori_img[y][x] < 50 and label == 0) then
				other_img[y][x] = 0
			end
		end
	end
	-- filter out noise in equal_top_img, equal_bottom_img and fraction_img
	-- fist blur and then threshold
	-- cv.imshow{"tmp", equal_bottom_img}
	-- cv.waitKey{0}
	--[[
	fraction_img = cv.boxFilter{ src=fraction_img, ksize={3,3}, ddepth=-1 }
	equal_top_img = cv.boxFilter{ src=equal_top_img, ksize={3,3}, ddepth=-1 }
	equal_bottom_img = cv.boxFilter{ src=equal_bottom_img, ksize={3,3}, ddepth=-1 }
	cv.threshold{fraction_img, fraction_img, 125, 255, cv.THRESH_BINARY}
	cv.threshold{equal_top_img, equal_top_img, 100, 255, cv.THRESH_BINARY}
	cv.threshold{equal_bottom_img, equal_bottom_img, 100, 255, cv.THRESH_BINARY}
	]]
	-- cv.imshow{"tmp", equal_bottom_img}
	-- cv.waitKey{0}
	cv.threshold{ori_img, ori_img, 50, 255, cv.THRESH_BINARY}
	return true
end

function rect_distance(r1, r2)
	local h_overlap, v_overlap
	if (r1.width > r2.width) then
		h_overlap = ((r2.x >= r1.x) and (r2.x <= r1.x + r1.width)) or ((r2.x + r2.width >= r1.x) and (r2.x + r2.width <= r1.x + r1.width))
	else
		h_overlap = ((r1.x >= r2.x) and (r1.x <= r2.x + r2.width)) or ((r1.x + r1.width >= r2.x) and (r1.x + r1.width <= r2.x + r2.width))
	end
	if (r1.height > r2.height) then
		v_overlap = ((r2.y >= r1.y) and (r2.y <= r1.y + r1.height)) or ((r2.y + r2.height >= r2.y) and (r2.y + r2.height <= r1.y + r1.height))
	else
		v_overlap = ((r1.y >= r2.y) and (r1.y <= r2.y + r2.height)) or ((r1.y + r1.height >= r1.y) and (r1.y + r1.height <= r2.y + r2.height))
	end
	if (h_overlap and v_overlap) then
		return 0
	end
	if (v_overlap) then
		return math.min(math.abs(r1.x - (r2.x + r2.width)), math.abs(r1.x + r1.width - r2.x))
	end
	if (h_overlap) then
		return math.min(math.abs(r1.y - (r2.y + r2.height)), math.abs(r1.y + r1.height - r2.y))
	end
	return math.min(math.abs(r1.x - (r2.x + r2.width)), math.abs(r1.x + r1.width - r2.x)) + math.min(math.abs(r1.y - (r2.y + r2.height)), math.abs(r1.y + r1.height - r2.y))
end

function normal_line_combine(r1, r2)
	local h_overlap = math.max(r1.x + r1.width, r2.x + r2.width) - math.min(r1.x, r2.x) - r1.width - r2.width
	if (h_overlap < 0) then
		h_overlap = -h_overlap
	else
		h_overlap = 0
	end
	if (h_overlap > 5) then
		return false
	end
	local v_overlap = math.max(r1.y + r1.height, r2.y + r2.height) - math.min(r1.y, r2.y) - r1.height - r2.height
	if (v_overlap < 0) then
		v_overlap = -v_overlap
	else
		v_overlap = 0
	end
	local ratio_1 = v_overlap / r1.height
	local ratio_2 = v_overlap / r2.height
	return (ratio_1 > 0.75 and ratio_2 > 0.75) or (ratio_1 > 0.9) or (ratio_2 > 0.9)
end


function segments_extraction(cur_line, line_idx)
	local size =  cur_line:size()
	local height = size[1]
	local width = size[2]
	local space_col_num = 0
	local start_col = 0
	local seg_idx = 1
	local space_interval_threshold = 3
	local col_sum = torch.sum(cur_line, 1)
	for cur_col = 1,width do
		-- if (torch.sum(line[cur_col]) == 255 * height) then
		if (col_sum[1][cur_col] == 255 * height) then
			-- this is a space column
			space_col_num = space_col_num + 1
			if (space_col_num == space_interval_threshold and start_col > 0) then
				-- get new line, from start_row to last_row
				local seg = cur_line:sub(1, height, start_col, cur_col - space_col_num)
				cv.imwrite { "lines/" .. filename_prefix .. "/" .. line_idx .. "_" .. seg_idx .. ".jpg", seg }
				seg_idx = seg_idx + 1
			end
		else
			-- this is not a space column
			if (space_col_num >= space_interval_threshold) then
				start_col = cur_col
			end
			space_col_num = 0
		end
	end
	if (space_col_num < space_interval_threshold) then
		local seg = cur_line:sub(1, height, start_col, width - space_col_num)
		cv.imwrite { "lines/" .. filename_prefix .. "/" .. line_idx .. "_" .. seg_idx .. ".jpg", seg }
	end
end

function compress_line(line)
	local size_ary = line:size()
	local height = size_ary[1]
	local width = size_ary[2]
	local sel_col_idx_ary = { }
	local sel_col_num = 1
	local space_col_thresh = 3
	local space_col_num = space_col_thresh - 2
	for c = 1, width do
		local sel = false
		if (line:sub(1, height, c, c):sum() == 255 * height) then
			if (space_col_num >= space_col_thresh) then
				sel = false
			else
				space_col_num = space_col_num + 1
				sel = true
			end
		else
			sel = true
			space_col_num = 0
		end
		if (sel == true) then
			sel_col_idx_ary[sel_col_num] = c
			sel_col_num = sel_col_num + 1
		end
	end
	return line:index(2, torch.LongTensor(sel_col_idx_ary))
end

equal_dilate_size = 8
fraction_dilate_size = 21

function main()
	for label_filename in lfs.dir(data_path .. "labels/") do
		if (label_filename ~= "." and label_filename ~= "..") then
			print(label_filename)
			local temp_idx = string.find(label_filename, "_label") - 1
			filename_prefix = label_filename:sub(1, temp_idx)

			-- the input of one image includes { original image, top equal sub-image, bottom equal sub-image, fraction sub-image }
			load_label_img(label_filename)

			-- dilate on the equal_top_img
			e = cv.getStructuringElement{shape=cv.MORPH_RECT, ksize={1, equal_dilate_size}}
			dilate_equal_top_img = cv.erode {src=equal_top_img, dst=nil, kernel=e, anchor={-1,equal_dilate_size - 1}, borderValue={255,255,255,255}}

			-- dilate on the equal_bottom_img
			-- e = cv.getStructuringElement{shape=cv.MORPH_RECT, ksize={1, equal_dilate_size}}
			dilate_equal_bottom_img = cv.erode {src=equal_bottom_img, dst=nil, kernel=e, anchor={-1,0}, borderValue={255,255,255,255}}

			-- dilate on the fraction_img
			e = cv.getStructuringElement{shape=cv.MORPH_RECT, ksize={1, fraction_dilate_size}}
			dilate_fraction_img = cv.erode {src=fraction_img, dst=nil, kernel=e, anchor={-1,-1}, borderValue={255,255,255,255}}

			dialate_equal_img = cv.addWeighted{dilate_equal_top_img, 0.5, dilate_equal_bottom_img, 0.5, 0}
			other_fraction_img = cv.addWeighted{other_img, 0.5, dilate_fraction_img, 0.5, 0}
			dilate_img = cv.addWeighted{dialate_equal_img, 0.5, other_fraction_img, 0.5, 0}
			dilate_thresh_img = dilate_img:clone()
			cv.threshold{dilate_img, dilate_thresh_img, 250, 255, cv.THRESH_BINARY}
			local for_line_label_img = line_extraction_2(dilate_thresh_img, ori_img, filename_prefix)

			-- labeling the results
			local line_label_img = torch.IntTensor(dilate_thresh_img:size())
			local label_num = cv.connectedComponents{for_line_label_img, line_label_img, 4}

			line_label_img = line_label_img:byte()
			rects = { }
			non_zero_num = { }
			lines = { }
			for i = 1, label_num - 1 do
				local cur_line_label_img = torch.ne(torch.eq(line_label_img, i), 1) * 255
				cur_line = cv.addWeighted{cur_line_label_img, 0.5, ori_img, 0.5, 0}
				cur_line = torch.eq(cur_line, 0)
				lines[i] = cur_line * 255
				rects[i] = cv.boundingRect{cur_line}
				non_zero_num[i] = torch.nonzero(cur_line):size()[1]
			end

			-- combine lines
			--- 1. find tiny lines and combine
			local tiny_num_threshold = 30
			local tiny_height_threshold = 5
			local ignore_num_threshold = 3
			tiny_lines = { }
			normal_lines = { }
			normal_lines_with_tiny = { }
			for i = 1, label_num - 1 do
				if (non_zero_num[i] > ignore_num_threshold) then
					if (non_zero_num[i] < tiny_num_threshold or rects[i].height < tiny_height_threshold) then
						table.insert(tiny_lines, i)
					else
						table.insert(normal_lines, i)
						table.insert(normal_lines_with_tiny, {i})
					end
				end
			end
			for i = 1, table.getn(tiny_lines) do
				local min_dist = -1
				local min_idx = -1
				local min_dist_threshold = 50
				for j = 1, table.getn(normal_lines) do
					local cur_dist = rect_distance(rects[tiny_lines[i]], rects[normal_lines[j]])
					if (min_dist == -1 or cur_dist < min_dist) then
						min_dist = cur_dist
						min_idx = j
					end
				end
				if (min_dist < min_dist_threshold) then
					table.insert(normal_lines_with_tiny[min_idx], tiny_lines[i])
				end
			end

			--- 2. combine normal lines
			normal_lines_after_combine = { }
			normal_lines_combine_index = { }
			local cluster_num = 0
			for i = 1, table.getn(normal_lines) do
				local find_cluster = false
				for c = 1, cluster_num do
					for j = 1, table.getn(normal_lines_after_combine[c]) do
						if (normal_line_combine(rects[normal_lines[i]], rects[normal_lines_after_combine[c][j]])) then
							find_cluster = true
							table.insert(normal_lines_after_combine[c], normal_lines[i])
							normal_lines_combine_index[normal_lines[i]] = c
							break
						end
					end
					if (find_cluster) then
						break
					end
				end
				if (find_cluster == false) then
					cluster_num = cluster_num + 1
					normal_lines_after_combine[cluster_num] = { normal_lines[i] }
					normal_lines_combine_index[normal_lines[i]] = cluster_num
				end
			end
			for i = 1, table.getn(normal_lines_with_tiny) do
				if (table.getn(normal_lines_with_tiny[i]) > 1) then
					local normal_line_idx = normal_lines_with_tiny[i][1]
					local cluster_idx = normal_lines_combine_index[normal_line_idx]
					for j = 2, table.getn(normal_lines_with_tiny[i]) do
						table.insert(normal_lines_after_combine[cluster_idx], normal_lines_with_tiny[i][j])
					end
				end
			end

			local m = nn.Sequential()
			m:add(nn.Padding(1, 5, 2, 255)):add(nn.Padding(1, -5, 2, 255)):add(nn.Padding(2, 5, 2, 255)):add(nn.Padding(2, -5, 2, 255))
			local final_lines = { }
			local final_line_locations = { }
			final_lines_local = { }
			for c = 1, table.getn(normal_lines_after_combine) do
				local temp = lines[normal_lines_after_combine[c][1]]
				for i = 2, table.getn(normal_lines_after_combine[c]) do
					temp = cv.addWeighted{temp, 0.5, lines[normal_lines_after_combine[c][i]], 0.5, 0}
				end
				final_lines[c] = torch.ByteTensor(temp:size()):fill(255) - torch.ne(temp, 0) * 255
				local rect = cv.boundingRect{temp}
				final_lines_local[c] = final_lines[c]:sub(rect.y + 1, rect.y + rect.height - 1, rect.x + 1, rect.x + rect.width - 1)
				final_lines_local[c] = m:forward(final_lines_local[c])
				final_line_locations[c] = rect
				-- cv.imshow{"final_line", final_lines_local[c]}
				-- cv.waitKey {0}
				lfs.mkdir("lines/" .. filename_prefix)
				cv.imwrite { "lines/" .. filename_prefix .. "/" .. c .. ".jpg", final_lines_local[c] }
				cv.imwrite { "lines/" .. filename_prefix .. "/" .. c .. "_compress.jpg", compress_line(final_lines_local[c]) }

				-- separate line into segments
				segments_extraction(final_lines_local[c], c)
			end
		end
	end
end


function tidyup()
	for line_dir in lfs.dir("lines/") do
		if (line_dir ~= "." and line_dir ~= "..") then
			-- print(line_dir)
			for line_file in lfs.dir("lines/" .. line_dir .. "/") do
				if (string.find(line_file, "compress") ~= nil) then
					local cmd = "cp " .. "lines/" .. line_dir .. "/" .. line_file .. " compressed_lines/" .. line_dir .. "_" .. line_file
					-- print(cmd)
					os.execute(cmd)
				end
			end
		end
	end
end


function line_extraction_jiafa_data()
	img_name_table = { }
	for img_name in lfs.dir("jiafa_data/") do
		if (img_name ~= "." and img_name ~= "..") then
			local spec_ary = mysplit(img_name, ".")
			local idx = tonumber(spec_ary[1])
			img_name_table[idx] = img_name
		end
	end
	line_idx = 1
	for idx = 1, table.getn(img_name_table) do
		img_file = cv.imread { "jiafa_data/" .. img_name_table[idx], cv.IMREAD_GRAYSCALE }
		for_line_label_img = line_extraction_2(img_file, img_file, idx)

		-- labeling the results
		line_label_img = torch.IntTensor(img_file:size())
		label_num = cv.connectedComponents{for_line_label_img, line_label_img, 4}
		line_label_img = line_label_img:byte()
		print(label_num)

		lines = { }
		non_zero_num = { }
		rects = { }
		for i = 1, label_num - 1 do
			cur_line_label_img = torch.ne(torch.eq(line_label_img, i), 1) * 255
			cur_line = cv.addWeighted{cur_line_label_img, 0.5, img_file, 0.5, 0}
			cur_line = torch.eq(cur_line, 0)
			lines[i] = cur_line * 255
			non_zero_num[i] = torch.nonzero(cur_line):size()[1]
			rects[i] = cv.boundingRect{cur_line}
			-- cv.imshow { "tmp", lines[i] }
			-- cv.waitKey {0}
		end


		-- find tiny lines and combine
		local tiny_num_threshold = 30
		local tiny_height_threshold = 5
		local ignore_num_threshold = 3
		tiny_lines = { }
		normal_lines = { }
		normal_lines_with_tiny = { }
		for i = 1, label_num - 1 do
			if (non_zero_num[i] > ignore_num_threshold) then
				if (non_zero_num[i] < tiny_num_threshold or rects[i].height < tiny_height_threshold) then
					table.insert(tiny_lines, i)
				else
					table.insert(normal_lines, i)
					table.insert(normal_lines_with_tiny, {i})
				end
			end
		end
		for i = 1, table.getn(tiny_lines) do
			local min_dist = -1
			local min_idx = -1
			local min_dist_threshold = 50
			for j = 1, table.getn(normal_lines) do
				local cur_dist = rect_distance(rects[tiny_lines[i]], rects[normal_lines[j]])
				if (min_dist == -1 or cur_dist < min_dist) then
					min_dist = cur_dist
					min_idx = j
				end
			end
			if (min_dist < min_dist_threshold) then
				table.insert(normal_lines_with_tiny[min_idx], tiny_lines[i])
			end
		end

		local m = nn.Sequential()
		m:add(nn.Padding(1, 5, 2, 255)):add(nn.Padding(1, -5, 2, 255)):add(nn.Padding(2, 5, 2, 255)):add(nn.Padding(2, -5, 2, 255))
		local final_lines = { }
		local final_line_locations = { }
		final_lines_local = { }
		for c = 1, table.getn(normal_lines_with_tiny) do
			local temp = lines[normal_lines_with_tiny[c][1]]
			for i = 2, table.getn(normal_lines_with_tiny[c]) do
				temp = cv.addWeighted{temp, 0.5, lines[normal_lines_with_tiny[c][i]], 0.5, 0}
			end
			final_lines[c] = torch.ByteTensor(temp:size()):fill(255) - torch.ne(temp, 0) * 255
			local rect = cv.boundingRect{temp}
			final_lines_local[c] = final_lines[c]:sub(rect.y + 1, rect.y + rect.height - 1, rect.x + 1, rect.x + rect.width - 1)
			final_lines_local[c] = m:forward(final_lines_local[c])
			final_line_locations[c] = rect
			-- cv.imshow{"final_line", final_lines_local[c]}
			-- cv.waitKey {0}
			cv.imwrite { "lines/" .. line_idx .. ".bmp", final_lines_local[c] }
			line_idx = line_idx + 1
		end
	end
end

line_extraction_jiafa_data()
