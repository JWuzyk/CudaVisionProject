import torch
import torch.nn as nn
import torchvision
from skimage.transform import resize

def create_dx_dy_gradient(height,width):
    dx_row = torch.arange(start=1, end = width+1, step = 1 , out=None, dtype=torch.float32)/float(width)
    dy_column = torch.arange(start=1, end = height+1, step= 1 , out=None, dtype=torch.float32)/float(height)
    dx = dx_row.unsqueeze(0).repeat(height,1)[None,None,:,:]
    dy = dy_column.unsqueeze(1).repeat(1,width)[None,None,:,:]
    return torch.cat((dx,dy),1)

class added_gradient(nn.Module):
    """
    This layer adds a bias with the specified width and height.
    The same bias is added across channel, i.e we broadcaste over 
    the channel dimension.
    If learnable is true, then the added bias will be learnable.
    If gradient is true, two more bias are added, each one with a
    scaling (from 0 to 1) along the x- and y-axis respectively.
    """
    def __init__(self,width,height,learnable=True,gradient=True):
        super(added_gradient, self).__init__()
        self.width = width
        self.height = height
        self.learnable = learnable 
        self.gradient = gradient
        if gradient:
            self.locationEncode = torch.cat((torch.ones(1,width,height),create_dx_dy_gradient(width,height)[0,:,:,:]), dim = 0)
        else:
            self.locationEncode = torch.ones(1,width,height)
            
        if learnable:
            if gradient:
                self.locationBias = torch.nn.Parameter(torch.zeros(3,width,height))
            else:
                self.locationBias = torch.nn.Parameter(torch.zeros(1,width,height))
                

    def forward(self,inputs):    
        if self.locationEncode.device != inputs.device:
            self.locationEncode=self.locationEncode.to(inputs.get_device())

        if self.learnable:
            if self.locationBias.device != inputs.device:
                self.locationBias = self.locationBias.to(inputs.get_device())
            if self.gradient:
                scaledLocationEncode = (self.locationEncode * self.locationBias)
                x = inputs + torch.sum(scaledLocationEncode, dim = 0)
            else:
                x = inputs + self.locationBias[0,:,:]
        else:
            x = inputs + torch.sum(self.locationEncode, dim = 0)
        return x

    def set_size(self, width,height):
        if self.gradient:
            self.locationEncode = torch.cat((torch.ones(1,width,height),create_dx_dy_gradient(width,height)[0,:,:,:]), dim = 0)
        else:
            self.locationEncode = torch.ones(1,width,height)
            
        if self.learnable:
            device = self.locationBias.device
            if str(device) != 'cpu':
                locationBias_np = self.locationBias.detach().cpu().numpy().transpose((1,2,0))
            else:
                locationBias_np = self.locationBias.detach().numpy().transpose((1,2,0))
            locationBias_np_rescaled = resize(locationBias_np, (width,height),anti_aliasing=True, order = 3) #order=0 is nearest neighbour, 1=bi-linear, 2=bi-quadratic, 3=bi-cubic
            self.locationBias = nn.Parameter(torch.Tensor(locationBias_np_rescaled.transpose((2,0,1))).to(device=device))


class concated_gradient(nn.Module):
    """
    This layer concatenates channels with the specified width and height.
    If learnable is true, then the concatenated layer will be learnable.
    If gradient is true, two more channels are added, each one with a
    scaling (from 0 to 1) along the x- and y-axis respectively.
    """
    def __init__(self,width,height,learnable=True,gradient=True):
        super(concated_gradient, self).__init__()
        self.width = width
        self.height = height
        self.learnable = learnable 
        self.gradient = gradient
        
        if gradient:
            self.locationEncode = torch.cat((torch.ones(1,width,height),create_dx_dy_gradient(width,height)[0,:,:,:]), dim = 0)
        else:
            self.locationEncode = torch.ones(1,width,height)
            
        if learnable:
            if gradient:
                self.locationBias = torch.nn.Parameter(torch.randn(3,width,height))
            else:
                self.locationBias = torch.nn.Parameter(torch.randn(1,width,height))
                
    def forward(self,inputs):    
        if self.locationEncode.device != inputs.device:
            self.locationEncode=self.locationEncode.to(inputs.get_device())

        if self.learnable:
            if self.locationBias.device != inputs.device:
                self.locationBias = self.locationBias.to(inputs.get_device())
            if self.gradient:
                scaledLocationEncode = self.locationEncode * self.locationBias
                x = torch.cat((inputs,scaledLocationEncode.repeat(inputs.shape[0],1,1,1)),dim=1)
            else:
                x = torch.cat((inputs,self.locationBias.repeat(inputs.shape[0],1,1,1)),dim=1)
        else:
            x = torch.cat((inputs,self.locationEncode.repeat(inputs.shape[0],1,1,1)),dim=1) 
        return x

    def set_size(self, width,height):
        if self.gradient:
            self.locationEncode = torch.cat((torch.ones(1,width,height),create_dx_dy_gradient(width,height)[0,:,:,:]), dim = 0)
        else:
            self.locationEncode = torch.ones(1,width,height)
            
        if self.learnable:
            device = self.locationBias.device
            if str(device) != 'cpu':
                locationBias_np = self.locationBias.detach().cpu().numpy().transpose((1,2,0))
            else:
                locationBias_np = self.locationBias.detach().numpy().transpose((1,2,0))
            locationBias_np_rescaled = resize(locationBias_np, (width,height),anti_aliasing=True, order = 3) #order=0 is nearest neighbour, 1=bi-linear, 2=bi-quadratic, 3=bi-cubic
            self.locationBias = nn.Parameter(torch.Tensor(locationBias_np_rescaled.transpose((2,0,1))).to(device=device))
    

class Net(nn.Module):
    """
        concatenated_bias - Three possible arguments
        "shared" : concatenated_gradient is added before the final convolutions are applied.
                    the same layer is used in both heads (shared weights)
        "separate" : concatenated_gradient is added before the final convolutions are applied.
                    the different layers are used in both heads
        "None" : no concatenated_gradient is added
        
        added_bias - Three possible arguments
        "shared" : added_gradient is added after the final convolutions are applied.
                    the same layer is used in both heads (shared weights)
        "separate" : added_gradient is added after the final convolutions are applied.
                    the different layers are used in both heads
        "None" : no added_gradient is added
        
        concatenated_bias_learnable = False - Determines whether there are learnable weights associated with 
                concatenated_gradient.
    """
    def __init__(self,concatenated_bias="None", concatenated_bias_learnable = False, added_bias = "None", input_dimension = (320,480)):
        super(Net, self).__init__()
        self.w = int(input_dimension[0]/4)
        self.h = int(input_dimension[1]/4)
        if input_dimension[0] % 32 != 0:
            print("Warning: Input_dimensions[0] are not divisible by 32.")
        if input_dimension[1] % 32 != 0:
            print("Warning: Input_dimensions[1] are not divisible by 32.")
        
        self.concatenated_bias=concatenated_bias
        self.added_bias = added_bias
        
        resnet = torchvision.models.resnet18(pretrained=False)
        self.features1 = nn.Sequential(*list(resnet.children())[:-5])
        self.features2 = nn.Sequential(*list(resnet.children())[-5])
        self.features3 = nn.Sequential(*list(resnet.children())[-4])
        self.features4 = nn.Sequential(*list(resnet.children())[-3])

        self.convT1 = nn.ConvTranspose2d(512,256,3,2,1,1)
        self.bn1 = nn.BatchNorm2d(512)
        self.convT2 = nn.ConvTranspose2d(512,256,3,2,1,1)
        self.bn2 = nn.BatchNorm2d(512)
        self.convT3 = nn.ConvTranspose2d(512,128,3,2,1,1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv1d1 = nn.Conv2d(256,256,1)
        self.conv1d2 = nn.Conv2d(128,256,1)
        self.conv1d3 = nn.Conv2d(64,128,1)

        if self.concatenated_bias == "shared" or self.concatenated_bias == "separate":
            self.det = nn.Conv2d(256 + 3,3,1)
            self.seg = nn.Conv2d(256 + 3,3,1)
        else: 
            self.det = nn.Conv2d(256,3,1)
            self.seg = nn.Conv2d(256,3,1)
            
        self.relu = nn.ReLU()

        if self.concatenated_bias == "shared":
            self.shared_bias = concated_gradient(self.w,self.h,learnable=concatenated_bias_learnable)
        if self.concatenated_bias == "separate":
            self.det_bias = concated_gradient(self.w,self.h,learnable=concatenated_bias_learnable)
            self.seg_bias = concated_gradient(self.w,self.h,learnable=concatenated_bias_learnable)
            
        if self.added_bias == "shared":
            self.shared_added_bias = added_gradient(self.w,self.h,learnable=True)
        if self.added_bias == "separate":
            self.det_added_bias = added_gradient(self.w,self.h,learnable=True)
            self.seg_added_bias = added_gradient(self.w,self.h,learnable=True)
            
    def freeze_encoder(self):
        for param in self.named_parameters():
            if param[0][:8] == "features":
                param[1].requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.named_parameters():
            if param[0][:8] == "features":
                param[1].requires_grad = True
    
    def forward(self, x):
      # Encode
        x1 = self.features1(x) # 
        x2 = self.features2(x1) # /2
        x3 = self.features3(x2) # /2
        x = self.features4(x3) # /2
      # Decode
        x = self.relu(x)
        x = self.convT1(x) # *2 256
        x3=self.conv1d1(x3) # 256
        x = torch.cat((x,x3),1) # 512

        x = self.relu(self.bn1(x)) 
        x = self.convT2(x) # *2 256
        x2= self.conv1d2(x2) # 256
        x = torch.cat((x,x2),1) # 512

        x = self.relu(self.bn2(x))
        x = self.convT3(x) # 128
        x1= self.conv1d3(x1) # 128
        x = torch.cat((x,x1),1) # 256

        x = self.relu(self.bn3(x))
        if self.concatenated_bias == "shared":
            x = self.shared_bias(x)
            x = (self.det(x),self.seg(x))
        if self.concatenated_bias == "separate":
            x = (self.det(self.det_bias(x)),self.seg(self.seg_bias(x)))
        if self.concatenated_bias == "None":
            x = (self.det(x),self.seg(x))

        if self.added_bias == "shared":
            output = (self.shared_added_bias(x[0]),self.shared_added_bias(x[1]))
        if self.added_bias == "separate":
            output = (self.det_added_bias(x[0]),self.seg_added_bias (x[1]))
        if self.added_bias == "None":
            output = x
            
        return output

    def set_input_dimension(self,input_dimension):
        self.w = int(input_dimension[0]/4)
        self.h = int(input_dimension[1]/4)
        if input_dimension[0] % 32 != 0:
            print("Warning: Input_dimensions[0] are not divisible by 32.")
        if input_dimension[1] % 32 != 0:
            print("Warning: Input_dimensions[1] are not divisible by 32.")

        if self.added_bias == "shared":
            self.shared_added_bias.set_size(self.w,self.h)
        if self.added_bias == "separate":
            self.det_added_bias.set_size(self.w,self.h)
            self.seg_added_bias.set_size(self.w,self.h)

        if self.concatenated_bias == "shared":
            self.shared_bias.set_size(self.w,self.h)
        if self.concatenated_bias == "separate":
            self.det_bias.set_size(self.w,self.h)
            self.seg_bias.set_size(self.w,self.h)
