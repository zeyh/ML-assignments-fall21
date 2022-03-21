import numpy as np
import matplotlib.pyplot as plt


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      alpha - (float) The weighting to give current rewards in estimating Q. This 
        should range [0,1], where 0 means "don't change the Q estimate based on 
        current reward" 
      gamma - (float) This is the weight given to expected future rewards when 
        estimating Q(s,a). It should be in the range [0,1]. Here 0 means "
        don't incorporate estimates of future rewards into the reestimate of
        Q(s,a)"
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
        
      See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode and for definitions of alpha, gamma, epsilon 
        (http://incompleteideas.net/book/RLbook2020.pdf).  
    """

    def __init__(self, epsilon=0.2, alpha=.5, gamma=.5, adaptive=False):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. Note that this is a different formula
        for the step size than was used in MultiArmedBandits. Use an
        epsilon-greedy policy for action selection. Note that unlike the
        pseudocode, we are looping over a total number of steps, and not a
        total number of episodes. This allows us to ensure that all of our
        trials have the same number of steps--and thus roughly the same amount
        of computation time.

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
          - Use the provided self._get_epsilon function whenever you need to
            obtain the current value of epsilon.
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state

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
        state_action_values = np.zeros((env.observation_space.n, env.action_space.n)) 
        rewards = np.zeros(100) #A 1D sequence of averaged rewards of length 100.
        s = np.floor(steps / 100) #s steps
        
        observation = env.reset()
        #Loop for each step of episode:
        for t in range(steps): #timesteps
            env.render()
            epsilon = self._get_epsilon(t/steps)

            #choose A from S
            action = np.argmax(state_action_values[observation]) #choose_best_action()
            if np.random.random() < epsilon: 
                action = np.random.randint(env.action_space.n)
    
            observation_nxt, reward, done, info = env.step(action)  #Take action A, observe R, S'
            # Q(S, A) +  alpha (R + y max_a Q(S', a) - Q(S, A))
            state_action_values[observation][action] += \
                self.alpha*(reward \
                    + self.gamma*np.max(state_action_values[observation_nxt]) \
                        - state_action_values[observation][action])
            rewards[int(t/s)] += reward/s
            observation = observation_nxt

            if done:
                observation = env.reset()
                
        # print(state_action_values)
        # env.close()
        return state_action_values, rewards
            
    def plot(self, rewards1, rewards2):
        x = np.linspace(1, 100, 100)
        plt.plot(x, rewards1, label="epsilon=0.01")
        plt.plot(x, rewards2, label="epsilon=0.5")
        plt.title("average rewards over 10 trails")
        plt.legend()
        plt.savefig('my_plot4.png')
        

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state
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

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. Should be called by the fit
        function during each step.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        An adaptive policy for epsilon-greedy reinforcement learning. Returns
        the current epsilon value given the learner's progress. This allows for
        the amount of exploratory vs exploitatory behavior to change over time.

        See free response question 3 for instructions on how to implement this
        function.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        self.epsilon = 1 - progress

