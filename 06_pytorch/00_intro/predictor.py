
# predictor.py
# This class convert numerical output to class label.
#
import numpy as np

class ILSVRCPredictor():
    """
    obtain ILSVRC data label from output

    Attributes
    ----------
    class_index (int) : label_name (str)
            dictionary-type variables
    """

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        """
        obtain ILSVRC label with highest prob from "out". 

        Parameters
        ----------
        out : torch.Size([1, 1000])
            output from network

        Returns
        -------
        predicted_label_name : str
            most plausible class name
        """
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

    def predict_top5(self, out):
        """
        obtain ILSVRC label and prob. with highest prob from "out". 

        Parameters
        ----------
        out : torch.Size([1, 1000])
            output from network

        Returns
        -------
        predicted_label_name : str
            most plausible class name
        """
        out_scores=out.detach().numpy()[0]
        total=np.sum(np.exp(out_scores))
        p_score=np.exp(out_scores)/total*100
        top5id=np.argsort(-out_scores)[0:5]
        top5key=[str(n) for n in top5id]
        print('\n results                    [%]')
        print('---------------------------------')
        for key in top5key:
            obj=self.class_index[key][1]
            p=p_score[int(key)]
            print('{:<20}'.format(obj), '{:>.2f}'.format(p))
        
        top5p=p_score[top5id]
        #print("top5id= ", top5id) #show top5 ID
        #print("top5p= ",top5p)    #show top5 probabilities
        return top5id, top5p




