from your_code import GradientDescent, load_data, accuracy
import numpy as np
import sklearn.metrics 
import matplotlib.pyplot as plt

print('Starting example experiment')

train_features, _, train_targets, _ = load_data('synthetic')

indices = [5,1,2,4]
train_features = train_features[indices]
train_targets = train_targets[indices]
print(train_features, train_targets)

bias = np.linspace(-5.5, 0.5, 100)
learner = GradientDescent('0-1')

loss_log = []
for i in range(bias.shape[0]):
    params = np.ones((train_features.shape[1]))
    params = np.insert(params, params.shape[0], bias[i])
    learner.model = params
    
    # print("features",train_features.shape, train_features, train_targets)
    predictions = learner.predict(train_features)
    # print("model", learner.model, predictions)
    
    tmp_features = np.insert(train_features, train_features.shape[1], 1, axis=1)
    # print("!!! bias added: ",tmp_features)
    loss_log.append(learner.loss.forward(tmp_features, params, train_targets))

# print(loss_log)

y_values = loss_log
# y_values = y_values[y_values != 0]
x_values = bias
plt.plot(x_values, y_values)
plt.xlabel("Bias")
plt.ylabel("Loss Scores")
plt.title('Loss')
plt.show()
    

# tmp = np.insert(train_features, train_features.shape[1], 1.0, axis=1)
# print(learner.loss.forward(train_features, learner.model, train_targets))
print('Finished example experiment')
