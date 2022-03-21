from your_code import GradientDescent, load_data, accuracy
import numpy as np
import sklearn.metrics 
import matplotlib.pyplot as plt

print('Starting example experiment')

train_features, _, train_targets, _ = load_data('mnist-binary')
# print(train_features)

reg_params = [1e-3, 1e-2, 1e-1, 1, 10, 100]
epsilon = 0.001

y1 = []
y2 = []

for r in reg_params:
    #-----l1
    learner = GradientDescent(loss='squared', regularization="l1",
                            learning_rate=1e-5, reg_param=r)
    learner.fit(
        train_features, train_targets, batch_size=50, max_iter=2000, 
        isPrinting=True, isMinibatch=True, test_features=train_features, test_targets=train_targets)

    non_zero_param = learner.model[learner.model >= epsilon]
    y1.append(non_zero_param.shape[0])
    
    #-----l2
    learner2 = GradientDescent(loss='squared', regularization="l2",
                        learning_rate=1e-5, reg_param=r)
    learner2.fit(
        train_features, train_targets, batch_size=50, max_iter=2000, 
        isPrinting=True, isMinibatch=True, test_features=train_features, test_targets=train_targets)

    non_zero_param2 = learner2.model[learner2.model >= epsilon]
    y2.append(non_zero_param2.shape[0])
    
# y_values = y1
# y_values = y_values[y_values != 0]
x_values = ["1e-3", "1e-2", "1e-1", "1", "10", "100"]
plt.plot(x_values, y1, 'bo', label = "l1 regularizer") 
plt.plot(x_values, y2, 'ro', label = "l2 regularizer") 
plt.legend()
plt.xlabel("lambda")
plt.ylabel("non-zero values")
plt.title('Regularization Comparison')
plt.show()


# print(loss_log)

# y_values = loss_log
# # y_values = y_values[y_values != 0]
# x_values = bias
# plt.plot(x_values, y_values)
# plt.xlabel("Bias")
# plt.ylabel("Loss Scores")
# plt.title('Loss')
# plt.show()

print('Finished example experiment')
