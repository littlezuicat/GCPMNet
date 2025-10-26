import torch
import torch.nn as nn

# [修改点]：不再导入 SwinImageProcessor，而是导入 AutoImageProcessor
from transformers import AutoImageProcessor, SwinModel 

class SwinT_KernelGenerator(nn.Module):
    def __init__(self, 
                 num_groups=8, 
                 kernel_size=3,
                 model_name='microsoft/swin-tiny-patch4-window7-224', 
                 freeze_swin=True):
        
        super().__init__()
        
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.output_kernel_params = num_groups * (kernel_size ** 2)

        print(f"Loading Swin Auto Image Processor: {model_name}")
        
        # [修改点]：使用 AutoImageProcessor.from_pretrained
        # 它会自动帮您加载正确的处理器 (无论是叫 SwinImageProcessor 还是 SwinFeatureExtractor)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        print(f"Loading Swin Model: {model_name}")
        # (这行不变)
        self.swin_model = SwinModel.from_pretrained(model_name)
        
        if freeze_swin:
            print("Freezing Swin Transformer parameters.")
            for param in self.swin_model.parameters():
                param.requires_grad = False
                
        swin_output_dim = self.swin_model.config.hidden_size
        self.projection_head = nn.Linear(swin_output_dim, self.output_kernel_params)
        self.act = nn.Tanh()

    def forward(self, images: torch.Tensor):
        # ... (forward 方法完全不需要改变) ...
        device = self.swin_model.device
        inputs = self.processor(images, return_tensors='pt').to(device)
        
        with torch.set_grad_enabled(not self.swin_model.training):
            outputs = self.swin_model(**inputs)
            
        g_vision = outputs.pooler_output
        kernel_flat = self.projection_head(g_vision)
        kernel_flat_act = self.act(kernel_flat)
        kernel = kernel_flat_act.reshape(self.num_groups, 1, self.kernel_size**2, 1)

        return kernel