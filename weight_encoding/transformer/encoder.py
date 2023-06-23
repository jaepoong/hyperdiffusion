from torch import nn

from weight_encoding.transformer.block import EncoderLayer,PositionalEncoding,Weight_Split

class Encoder(nn.Module):
    
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device, down_layer=[2,4,6,7]):
        super().__init__()
        
        self.emb = PositionalEncoding(d_model, max_len, device)
        
        self.d_model = d_model
        
        self.ffn_hidden = ffn_hidden
        self.layers = nn.ModuleList()
        self.n_head = n_head
        self.out = d_model
        for i in range(n_layer):
            if i in down_layer:
                self.out = self.d_model//2
            
            self.layers.append(EncoderLayer(d_model=self.d_model,
                                            ffn_hidden=self.ffn_hidden,
                                            n_head = self.n_head,
                                            drop_prob = drop_prob,
                                            out = self.out
                                            ))
            if i in down_layer:
                self.d_model=self.d_model//2
                self.ffn_hidden=self.ffn_hidden//2
            
            
    
    def forward(self, x, src_mask=None):
        
        x = self.emb(x) # [batch, len, d_model]
        
        for layer in self.layers:
            x = layer(x,src_mask)
        
        return x
        
        
        
    
        
        
    
    