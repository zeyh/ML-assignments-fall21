import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed. Use an epsilon-greedy policy for action selection.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        s = env.observation_space.n # the number of states in the environment -> 1 state and 10 actions
        a = env.action_space.n # the number of possible actions #10 slot machines
        state_action_values = np.zeros((s, a)) 
        rewards = np.zeros(100) #A 1D sequence of averaged rewards of length 100.
        s = np.floor(steps / 100) #s steps
        num_action = np.zeros(a)
        observation = env.reset() #curr state
        
        for t in range(steps): #timesteps
            env.render()
            isRandom = np.random.choice(2, size=1, p=[1-self.epsilon, self.epsilon])
            action = np.argmax(state_action_values[observation])
            if isRandom: 
                action = env.action_space.sample()
                
            observation, reward, done, info = env.step(action) 
            num_action[action] += 1
            state_action_values[observation][action] += (1/num_action[action]) * (reward - state_action_values[observation][action])
            # print("!!",observation, reward, done, info)
            rewards[int(t/s)] += reward/s
            
        env.close()
        return state_action_values, rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.
          - You should use a loop to predict over each step in an episode until
            it terminates; see /src/slot_machines.py for an example of how an
            environment signals the end of an episode using the step() method

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        observation = env.reset()
        states = [] #length K 
        actions = [] #the number of steps taken within the episode.
        rewards = []
        done = False
        while not done:
            action = np.argmax(state_action_values[observation])
            observation, reward, done, info = env.step(action)
            states.append(observation)
            actions.append(action)
            rewards.append(reward)
        # print(states, actions, rewards)
        
        return np.array(states), np.array(actions), np.array(rewards)
    
    def plot(self, rewards1, rewards2, rewards3):
        x = np.linspace(1, 100, 100)
        plt.plot(x, rewards1, label="only with first trial")
        plt.plot(x, rewards2, label="average between 5 trials")
        plt.plot(x, rewards3, label="average between 10 trials")
        plt.title("average rewards")
        plt.legend()
        plt.savefig('my_plot.png')
        
    
    if __name__ == "__main__":
        import gym
        print("testing")

    


