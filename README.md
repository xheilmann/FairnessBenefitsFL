## Benefits of the Federation? Analyzing the Impact of Fair Federated Learning at the Client Level
-----------------------------------------------------------------------------------

Federated Learning (FL) enables collaborative model training while preserving participating clients' local data privacy. However, the diverse data distributions across different clients can exacerbate fairness issues, as biases inherent in client data may propagate across the Federation. Although various approaches have been proposed to enhance fairness in FL, they typically focus on mitigating the bias of a single binary-sensitive attribute. This narrow focus often overlooks the complexity introduced by clients with conflicting or diverse fairness objectives. Such clients may contribute to the Federation without experiencing any improvement in their own model's performance or fairness regarding their specific sensitive attributes. In this paper, we compare three approaches to mitigate model unfairness in scenarios where clients have differing and potentially conflicting fairness requirements. By analyzing disparities across sensitive attributes and model performance, we investigate the conditions under which clients benefit from Federation participation. 
Our findings emphasize the importance of aligning Federation objectives with diverse client needs to enhance participation and equitable outcomes in FL settings.

### Repo Structure
We provide in the Repo a directory for each method we used in our comparision: 
* PUFFLE
* Reweighting
* FedMinMax
* Local and Global Models

The preprocessed input data The data can be found in `federated-fairness-main/preprocessed_data`.

### How to install the dependencies

For PUFFLE and Reweighting the Poetry dependency manager is used. If you don't have poetry installed you can run:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Then, you can install all the dependencies with:

- poetry install 

For FedminMax and the local and global models we procide a `requirements.txt` file in the respective directories. 

### How to run the code
As all methods we applied differ in how they we provide `run.py` files as well as some extended `README.md` from the original code files in each directory. 
