from torch import nn

from weight_encoding.transformer.block import DecoderLayer

class Decoder(nn.Module):
    def __init__(self,device, d_model=1024 , ffn_hidden=3072 ,n_head=16, n_layer=20, drop_prob=0.1, input = 1024):
        super().__init__()
        self.d_model=d_model
        self.layers=nn.ModuleList()
        self.n_head=n_head
        
        self.layers.append(DecoderLayer(d_model = input,
                                        ffn_hidden= ffn_hidden,
                                        n_head=self.n_head,
                                        drop_prob=drop_prob,
                                        out=d_model
            
        ))
        
        for i in range(n_layer):
            self.layers.append(DecoderLayer(d_model=self.d_model,
                                                      ffn_hidden=ffn_hidden,
                                                      n_head=self.n_head,
                                                      drop_prob=drop_prob,
                                                      out=self.d_model
                                                      ))
        
    def forward(self, enc,dec):
        x=dec
        for layer in self.layers:
            x=layer(x,x)

        return x

"""class Decoder(nn.Module):
    def __init__(self, d_model , ffn_hidden ,n_head, n_layer, drop_prob, device, up_layer=[]): #[2,4,6,8]
        super().__init__()
        self.d_model=d_model
        self.layers=nn.ModuleList()
        self.n_head=n_head
        self.out = self.d_model//(2**len(up_layer))
        self.ffn_hidden = ffn_hidden//(2**len(up_layer))
        self.d_model = self.d_model//(2**len(up_layer))
        
        
        for i in range(n_layer):
            if i in up_layer:
                self.out=self.out*2
                self.ffn_hidden = self.ffn_hidden*2
            self.layers.append(DecoderLayer(d_model=self.d_model,
                                                      ffn_hidden=self.ffn_hidden,
                                                      n_head=self.n_head,
                                                      drop_prob=drop_prob,
                                                      out=self.out
                                                      ))
            if i in up_layer:
                self.d_model = self.d_model*2
        
    def forward(self, enc,dec):
        x=dec
        for layer in self.layers:
            x=layer(x,x)

        return x"""