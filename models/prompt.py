import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextPromptEncoder(nn.Module):
    """
    一个用于将文本提示（Text Prompts）编码为向量的模块。
    它内部封装了BERT模型和Tokenizer。
    """
    def __init__(self, 
                 model_name='bert-base-uncased', 
                 output_dim=256, # 注意：您原来的代码并未使用此参数
                 freeze_bert=True,
                 group=8):
        """
        初始化函数
        参数:
        - model_name (str): 你想使用的预训练BERT模型。
          (如果你的提示是中文，可以使用 'bert-base-chinese')
        - output_dim (int): 你希望最终输出的向量维度，用于匹配你的视觉模型。
        - freeze_bert (bool): 是否冻结BERT的权重。强烈建议初次训练时设为 True！
        """
        super().__init__()
        self.group = group
        print(f"Loading BERT Tokenizer: {model_name}")
        # 1. 加载 Tokenizer
        # add_special_tokens=True 会自动添加 [CLS] 和 [SEP]
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        print(f"Loading BERT Model: {model_name}")
        # 2. 加载预训练的BERT模型 (只加载基础模型，不要分类头)
        self.bert_model = BertModel.from_pretrained(model_name)
        
        # 3. (可选但强烈推荐) 冻结BERT的参数
        if freeze_bert:
            print("Freezing BERT parameters.")
            for param in self.bert_model.parameters():
                param.requires_grad = False
                
        # 4. BERT的输出维度通常是768 (bert-base-*)
        bert_output_dim = self.bert_model.config.hidden_size # (例如 768)
        
        # 5. 一个“投影头”(Projection Head)，将768维映射到你需要的维度
        #    (您原来的代码是映射到 group*3*3)
        self.projection_head = nn.Linear(bert_output_dim, group*3*3)
        
    def forward(self, text_prompt: str): # <--- [修改点 1]：从 list[str] 变为 str
        """
        前向传播
        参数:
        - text_prompt (str): 一个单独的文本字符串。
          例如: "a cityscape with dense haze"
        """
        
        # 0. 确定设备 (确保tokenizer的输出和模型在同一设备上)
        device = self.bert_model.device
        
        # 1. Tokenization (分词与编码)
        #    当处理单个字符串时，不需要 padding=True
        #    return_tensors='pt' 会自动创建一个 [1, sequence_length] 的张量
        inputs = self.tokenizer(
            text_prompt,              # <--- [修改点 2]：使用 text_prompt
            # padding=True,         # <--- [修改点 2]：移除 padding
            truncation=True, 
            return_tensors='pt'
        )
        
        # 2. 将tokenizer的输出移动到GPU/CPU
        input_ids = inputs['input_ids'].to(device)       # Shape: [1, seq_len]
        attention_mask = inputs['attention_mask'].to(device) # Shape: [1, seq_len]
        
        # 3. 通过BERT模型
        #    模型仍然需要一个 [batch, seq_len] 的输入，我们这里的 batch_size 是 1
        with torch.set_grad_enabled(not self.bert_model.training): # 如果冻结了，这里不计算梯度
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # 4. 提取 [CLS] Token 的 Embedding
        # 'last_hidden_state' 的形状是 [1, sequence_length, 768]
        # 我们只取第一个Token (即 [CLS] Token) 的向量
        cls_embedding = outputs.last_hidden_state[:, 0] # Shape: [1, 768]
        
        # 5. 通过投影头，得到最终的Embedding
        projection_output = self.projection_head(cls_embedding) # Shape: [1, group*3*3]
        
        # 6. Reshape 并移除 batch 维度
        #    原来的 reshape(-1, ...) 中的 -1 会被解析为 1 (batch_size)
        #    我们直接去掉 -1，将其 reshape 为目标形状
        final_embedding = projection_output.reshape(self.group, 1, 3*3, 1) # <--- [修改点 3]
        # Shape: [group, 1, 9, 1]
    
        
        return final_embedding