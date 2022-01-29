import torch.nn as nn
from graphLayer import GCU
import torch
from config import args

class GraphConv2(nn.Module):
    def __init__(self, batch = 1, h=[16,32,64,128,256], w=[16,32,64,128,256], d=[768,512], V=[2,4,8,32],outfeatures=[64,32]):
        super(GraphConv2, self).__init__()

        self.gc1 = GCU(batch = batch, h=h[0], w=w[0], d=d[0], V=V[0],outfeatures=outfeatures[0])
        self.gc2 = GCU(batch = batch, h=h[0], w=w[0], d=d[0], V=V[1],outfeatures=outfeatures[0])
        self.gc3 = GCU(batch = batch, h=h[0], w=w[0], d=d[0], V=V[1],outfeatures=outfeatures[0])
        self.gc4 = GCU(batch = batch, h=h[0], w=w[0], d=d[0], V=V[3],outfeatures=outfeatures[0])
      
    def forward(self, x):
        
        y = self.gc3(x)
        out = torch.cat((x,y),dim=1)

        return out