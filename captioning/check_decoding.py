"""Synthetic transition checks for EOS handling, beam selection and batch isolation."""
import torch
from decoding_search import search_features


class Toy:
    def start_state(self, spatial):
        n=spatial.shape[0]
        return (torch.zeros(n,1),torch.zeros(n,1))

    def step(self, token, spatial, projected, state):
        logits=torch.full((len(token),7),-30.)
        for i,t in enumerate(token.tolist()):
            if t==1:
                logits[i,4]=0.4;logits[i,5]=0.
            elif t==4:
                logits[i,2]=0.;logits[i,6]=0.
            elif t==5:
                logits[i,2]=5.
            else:
                logits[i,2]=5.
        # Distinct image type whose preferred text is a single token 6.
        mask=spatial[:,0,0].eq(1)&token.eq(1)
        logits[mask]=-30.;logits[mask,6]=10.
        return logits,state


features=torch.tensor([[[0.]],[[1.]]])
greedy=search_features(Toy(),features,features,beam_size=1,alpha=0.,constrain=True)
beam=search_features(Toy(),features,features,beam_size=3,alpha=0.,constrain=True)
assert greedy[0,0].item()==4
assert beam[0,:2].tolist()==[5,2]  # globally better sequence than greedy
assert beam[1,:2].tolist()==[6,2]  # parents from another image never mix
assert all(t==2 for t in beam[0,1:].tolist())
print('PASS: beam beats greedy on toy sequence, batch isolation, EOS score preservation')
