from torch import nn
import math
import torch

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,q,k,v,mask=None,e=1e-12):
        
        batch_size,head,length,d_tensor=k.size()
        # dot product of query and keys
        k_t=k.transpose(2,3)
        score=(q @ k_t)/ math.sqrt(d_tensor)
        
        if mask is not None:
            score=score.masked_fill(mask==0,-10000)
        
        score=self.softmax(score)
        
        v=score @ v
        
        return v, score


class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.n_head=n_head
        self.attention=ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        
        self.w_concat = nn.Linear(d_model,d_model)
    
    def forward(self, q, k , v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # split to multi head
        q,k,v= self.split(q), self.split(k), self.split(v)
        
        # do scaledotproduct attention, outpug -> v, score :: score is output of query key dot-product.
        out,attention = self.attention(q,k,v,mask=mask)
        
        out = self.concat(out)
        out = self.w_concat(out)
        
        return out
        

    def split(self,tensor):
        ''' 
            split tensor by number of head
            param tensor : [batch, length, d_model]
            return : [batch,head,length,d_tensor]
        '''
        
        batch,length,d_model = tensor.size()
        d_tensor = d_model //self.n_head
        tensor = tensor.view(batch,length,self.n_head,d_tensor).transpose(1,2)
        
        return tensor
    
    def concat(self,tensor):
        '''
            concat splited tensor to original length
        '''
        
        batch,head,length,d_tensor= tensor.size()
        d_model=head * d_tensor
        
        tensor = tensor.transpose(1,2).contiguous().view(batch,length,d_model)
        return tensor
    


class LayerNorm(nn.Module):
    def __init__(self,d_model, eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        self.eps=eps
    
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        var = x.var(-1,keepdim=True)
        
        out=(x-mean) / torch.sqrt(var + self.eps) # normalizing
        
        out = self.gamma * out + self.beta
        
        return out
    


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, out, drop_prob=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden, out) # changed to output dimmension to fluent
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x=self.linear1(x)
        x=self.relu(x)  # skip-connection
        x=self.dropout(x)
        x=self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model,ffn_hidden, n_head,drop_prob, out=None):
        super(EncoderLayer,self).__init__()
        
        self.out = out if out else d_model # output size를 define하면 그대로 아니면 원래 d_model사이즈 그대로 받음. for down or upsampling
        
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        
        self.norm1 = LayerNorm(d_model)
        self.dropout1=nn.Dropout(drop_prob)
        
        self.ffn= PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, out=self.out, drop_prob=drop_prob)
        
        self.norm2 = LayerNorm(d_model=self.out)
        self.dropout2=nn.Dropout(drop_prob)
    
    
    def forward(self, x, src_mask):
        _x = x
        # multi head attention
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        
        # position wise feedforward
        
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x)
        
        return x
        
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head, drop_prob, out=None):
        
        super(DecoderLayer, self).__init__()
        self.out = out if out else d_model
        
        self.self_attention = MultiHeadAttention(d_model, n_head=n_head)
        
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(p=drop_prob)
        
        self.enc_dec_attention=MultiHeadAttention(d_model = d_model,n_head = n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2=nn.Dropout(p=drop_prob)
        
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob,out=self.out)
        self.norm3=LayerNorm(d_model=out)
        self.dropout3=nn.Dropout(p=drop_prob)
    
    def forward(self,dec,enc):
        _x=dec
        # self attention
        x = self.self_attention(q=dec,k=dec,v=dec)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        
        # encoder_decoder attention
        
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x,k=enc,v=enc)
            x = self.dropout2(x)
            x = self.norm2(x+_x)
        
        x = self.ffn(x)
        
        x = self.dropout3(x)
        x = self.norm3(x)
        return x
        
        

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding,self).__init__()
        self.device=device
        self.encoding = torch.zeros(max_len, d_model, device = device)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0,max_len,device=device)
        pos = pos.float().unsqueeze(dim=1)
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:,0::2] = torch.sin(pos/(10000 **(_2i / d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i / d_model)))
    
    def forward(self, x):
        batch,seq_len,d = x.size()
        
        return self.encoding[:seq_len, :].unsqueeze(0) + x
        

class Weight_Split(nn.Module):
    def __init__(self, input_parameter_sizes, output_parameter_sizes, #input_parameter_names,
                 split_policy='chunk',chunk_size=512
                 ):
        '''
            input_parameter_sizes : 파라미터의 크기를 입력할것
            output_parameter_sizes : 파라미터의 크기를 입력할것
        '''
        super().__init__()
    
        self.input_splits=self.build_splits(input_parameter_sizes,chunk_size=chunk_size)
        self.output_splits=self.build_splits(output_parameter_sizes,chunk_size=chunk_size)
    
    @staticmethod
    def build_splits(parameter_sizes,split_policy ='chunk',chunk_size=None):
        
        if split_policy=="chunk":
            total_n_params=sum(parameter_sizes)
            num=total_n_params//chunk_size
            print(num)
            splits=[chunk_size]*num
            remainder=total_n_params%chunk_size
            if remainder>0:
                splits.append(remainder)
        
        return splits

    def encode_parameters(self, parameters):
        """
        입력 파라미터 self.input_splits사용 쪼개기.
        """
        split_parameters=torch.split(parameters, self.input_splits, dim=1)
        return torch.stack(split_parameters,dim=1)
    
    def forward(self,x):
        embeddings=self.encode_parameters(x)
        b,t,d = embeddings.size()
        
        return embeddings