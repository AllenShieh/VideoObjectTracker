import pafy
import csv
import cv2

#url = "https://www.youtube.com/watch?v=AAB6lO-XiKE"
#video = pafy.new(url)
#best = video.getbest(preftype="webm")
#best.download(quiet=False)

reader = csv.reader(open("youtube_boundingboxes_detection_train.csv","r"))

result = []
c = 0
for item in reader:
  if item[0]!='AAB6lO-XiKE':
    break
  #print(item)
  result.append(item)

#print result

cap = cv2.VideoCapture("Abertura das festividades Oktoberfest 2010 Munique.webm")

