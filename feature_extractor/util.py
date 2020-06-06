import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
import cv2



class CommonFeatureExtractor(object):
    __instance = None

    @staticmethod
    def get_instance():
        if CommonFeatureExtractor.__instance == None:
            CommonFeatureExtractor()
        return CommonFeatureExtractor.__instance

    def __init__(self):
        if CommonFeatureExtractor.__instance == None:
            CommonFeatureExtractor.__instance = self
        self.model = self.__load_model_cov()

    def __load_model_cov(self):
        model_conv = models.resnet18(pretrained=True)
        layer = model_conv._modules.get('avgpool')
        model_conv.eval()
        last_layer = nn.Sequential(*list(model_conv.children())[:-1])
        return last_layer

    def get_feature_vectors(self, cv_image):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv_image)
        test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        image_tensor = test_transform(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        output = self.model(inp)
        output = [float(i[0][0]) for i in list(output[0])]
        return output
