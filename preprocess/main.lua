local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'lfs'

for img_file in lfs.dir("300dpi") do
	print(img_file)
	if (img_file ~= "." and img_file ~= "..") then
		local im_gray = cv.imread { "300dpi/" .. img_file, cv.IMREAD_GRAYSCALE }
		local im_bw = cv.adaptiveThreshold{im_gray, im_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 21}
		cv.imwrite { "300dpi_binary/" .. img_file, im_bw }
	end
end


--[[
-- read image from file
local im_gray = cv.imread {"test.jpg", cv.IMREAD_GRAYSCALE}

-- do threshold
local im_bw
im_bw = cv.adaptiveThreshold{im_gray, im_bw, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 19}

-- save the binary image
cv.imwrite {"binary.jpg", im_bw}
]]


--[[
cv.imshow {"grayscale", im_gray}
cv.imshow {"binary", im_bw}
cv.waitKey {0}
]]
