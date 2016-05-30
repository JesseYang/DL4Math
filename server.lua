local cv = require 'cv'
require 'cv.imgproc'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'binarize/main'
require 'line_extraction/main'
require 'sequence_labeling/main'

local app = require('waffle')

app.post('/upload', function(req, res)
	-- 1. convert to ByteTensor
	img = req.form.file:toImage()
	img = (img * 255):byte()
	img = img:transpose(1, 2):transpose(2, 3)
	-- 2. convert to binary image
	cv.cvtColor{img, img, cv.COLOR_BGR2GRAY}
	img = img:transpose(2,3):transpose(1,2):sub(1,1,1,-1,1,-1)[1]
	img = binarize(img)
	-- 3. extract lines
	lines = extract_lines(img)
	for i = 1, table.getn(lines) do
		cv.imwrite { "line_" .. i .. ".jpg", lines[i] }
	end
	-- res.send("done")
	results = { }
	result_str = ""
	-- 4. scale and padding to 80 pixels high, and then recognize
	for n = 1, table.getn(lines) do
		preprocess_line = scale_and_padding(lines[n])
		results[n] = recognize(preprocess_line)
		result_str = result_str .. results[n] .. "\n"
	end
	print(result_str)
	res.send(result_str)
end)

app.get('/upload', function(req, res)
	res.render('upload.html', { })
end)

-- app.listen({host="10.50.101.163", port="8080"})
-- app.listen({host="127.0.0.1", port="8080"})
app.listen({host="10.8.0.4", port="8080"})
