##Benefits of the Federation? Analyzing the Impact of Fair Federated Learning at the Client Level
-----------------------------------------------------------------------------------

Federated Learning (FL) enables collaborative model training while preserving the privacy of participating clients’ local data. However,the diverse data distributions across the different clients can exacerbate fairness issues, as biases inherent in client data may propagate 
across the federation. Although various approaches have been proposed to enhance fairness in FL, they typically focus on mitigating the bias of a single binary-sensitive attribute. This narrow focus often overlooks the complexity introduced by clients with conflicting
or diverse fairness objectives. Such clients may contribute to the federation without experiencing any improvement in their own model’s performance or fairness regarding their specific sensitive attributes. In this paper, we compare three approaches to mitigate
model unfairness in scenarios where clients have differing and potentially conflicting fairness requirements. Through analysis of disparities across sensitive attributes and model performance, we investigate the conditions under which clients benefit from federation
participation. Our findings emphasize the importance of aligning federation objectives and communicating these with diverse client needs to enhance participation and equitable outcomes in FL settings.

We here provide the code and the preprocessed data for this analysis. The data can be found in `federated-fairness-main/preprocessed_data`. Each file provides their own requirements.txt file to run the code.
