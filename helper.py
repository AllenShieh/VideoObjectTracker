import pafy
import csv
#import cv2

#url = "https://www.youtube.com/watch?v=AAB6lO-XiKE"
#video = pafy.new(url)
#best = video.getbest(preftype="webm")
#best.download(quiet=False)

reader = csv.reader(open("youtube_boundingboxes_detection_validation.csv","r"))

result = []
previous = "";
c = 0
for item in reader:
  #print(item)
  if item[0] != previous:
      result.append(item[0])
      previous = item[0]

#print result
print len(result)

for i in range(10):
    url = "https://www.youtube.com/watch?v=" + result[i]
    video = pafy.new(url)
    best = video.getbest(preftype="webm")
    best.download(quiet=False)
