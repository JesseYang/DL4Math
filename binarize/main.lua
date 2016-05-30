local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'lfs'
require './util'
require 'image'

dpi = "150dpi"

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

function water_flow(im_bw, blur)
	print("111")
	local im_bw_left = im_bw:clone():fill(0)
	for i = 1, width do
		for j = 1, height do
			if (im_bw[j][i] == 255) then
				-- this is background point, should be decided whether wet or not
				local level_idx = torch.floor(blur[j][i] / 6) - 34
				if (level_idx < 1) then
					level_idx = 1
				end

				local angle_level = angle_level_ary[level_idx]
				if (i == 1) then
					-- background points in the first column are always wet 
					im_bw_left[j][i] = 122
				else
					if (im_bw_left[j][i-1] == 122) then
						im_bw_left[j][i] = 122
					else
						if (i > angle_level) then
							-- check the above row
							if (j > 1) then
								if (torch.sum(im_bw_left:sub(j-1, j-1, i-angle_level, i-1)) == 122 * angle_level) then
									im_bw_left[j][i] = 122
								end
							end
							-- check the below row
							if (im_bw_left[j][i] ~= 122 and j < height) then
								if (torch.sum(im_bw_left:sub(j+1, j+1, i-angle_level, i-1)) == 122 * angle_level) then
									im_bw_left[j][i] = 122
								end
							end
						end
					end
				end
			end
		end
	end
	return im_bw_left
end

function line_extraction_2(im_bw, line_dir_name)
	im_bw_blur = im_bw:clone()
	cv.GaussianBlur { src = im_bw, ksize={21, 21}, sigmaX=10, dst=im_bw_blur, sigmaY=10}

	angle_level_ary = { 1, 2, 3, 4, 5, 6, 7, 8 }
	size = im_bw:size()
	height = size[1]
	width = size[2]
	-- first water flow from left to right
	local im_bw_left = water_flow(im_bw, im_bw_blur)
	local im_bw_right = image.hflip(water_flow(image.hflip(im_bw), image.hflip(im_bw_blur)))

	im_bw_water = im_bw:clone()
	for i = 1, width do
		for j = 1, height do
			if (im_bw_left[j][i] == 122 and im_bw_right[j][i] == 122) then
				im_bw_water[j][i] = 122
			end
		end
	end
	cv.imwrite { line_dir_name .. "_water.jpg", im_bw_water }
end

function binarize_all()
	lfs.mkdir(dpi .. "_binary")

	for img_file in lfs.dir(dpi) do
		if (img_file ~= "." and img_file ~= "..") then
			print(img_file)

			-- binarization
			local im_gray = cv.imread { dpi .. "/" .. img_file, cv.IMREAD_GRAYSCALE }
			if (img_file:len() > 13) then
				img_file = img_file:sub(7)
			end
			-- local im_bw = cv.adaptiveThreshold{im_gray, im_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 21}
			local im_bw = binarize(im_gray)
			cv.imwrite { dpi .. "_binary/" .. img_file, im_bw }
		end
	end
end


function binarize(im_gray)
	local im_bw = cv.adaptiveThreshold{im_gray, im_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 21}
	return im_bw
end
