import scipy.io as sio

def loadInfo(filename):
  mat_contents = sio.loadmat(filename)
  frameInfo = mat_contents['frameInfo'][0]
  frameInfoList = []
  for i in range(len(frameInfo)):
    if len(frameInfo[i]) == 0:
      frameInfoList.append(None)
    else:
      frameInfoList.append(frameInfo[i][1])
  return frameInfoList