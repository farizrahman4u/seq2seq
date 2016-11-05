from zipfile import ZipFile
import os
import urllib
import urllib2


#Download GloVe word embeddings
url = 'http://nlp.stanford.edu/data/glove.6B.zip'
file_name = url.split('/')[-1]
u = urllib2.urlopen(url)
meta = u.info()
file_size = int(meta.getheaders("Content-Length")[0])
file_exists = False
if os.path.isfile(file_name):
	local_file_size = os.path.getsize(file_name)
	if local_file_size == file_size:
		file_exists = True
	else:
		print "File corrupt. Downloading again."
		os.remove(file_name)
if not file_exists:
	print "Downloading: %s Bytes: %s" % (file_name, file_size)
	file_size_dl = 0
	block_sz = 8192
	f = open(file_name, 'wb')
	while True:
	    buffer = u.read(block_sz)
	    if not buffer:
	        break
	    file_size_dl += len(buffer)
	    f.write(buffer)
	    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
	    status = status + chr(8)*(len(status)+1)
	    print status,
	f.close()
else:
	print "File already exists."
#Unzip file
print "Extracting zip..."
with ZipFile(file_name) as zip_file:
	zip_file.extractall()
print "Done."

