import ray,gym
import numpy as np
from random import randrange
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn import DQN
from scipy.stats import norm
from ray.tune.logger import pretty_print

saveData = True
saveId = 'trueMDP_trinomialModel_'+ str(1)
trainTime_epochs = 1000

env_config = {'globalEnv' : {'n_actions' : 21, 'n_samples' : 60, 'n_episodes' : 100000, 'trueMDP' : True},
                 'gbm' : {'mu' : 0.0, 'sigma' : 0.3, 'dt' : 1/365, 's0' : 100},
                 'opt' : {'strike' : 100, 'T' : 60/365}}


class OptionHedgingEnvMinimal(gym.Env):
    """This environment models an option market"""

    def __init__(self, envParameters, load = False):
        self.state = np.zeros(5)
        self.n_actions = envParameters['globalEnv']['n_actions']
        self.n_samples = envParameters['globalEnv']['n_samples']
        self.n_episodes = envParameters['globalEnv']['n_episodes']
        self.trueMDP = envParameters['globalEnv']['trueMDP'] #if False, then at each time the discrepancy between current hedge and BSM price is returned as reward. If True, a true MDP similar to Buehler at al.

        self.mu = envParameters['gbm']['mu']
        self.sigma = envParameters['gbm']['sigma']
        self.dt = envParameters['gbm']['dt']
        self.s0 = envParameters['gbm']['s0']

        self.upFactor = np.exp(self.sigma*np.sqrt(2*self.dt))
        self.downFactor = 1/self.upFactor
        tmpUp = np.exp(self.sigma*np.sqrt(0.5*self.dt))
        self.probUp = ((1-1/tmpUp)/(tmpUp-1/tmpUp))**2
        self.probDown = ((tmpUp-1)/(tmpUp-1/tmpUp))**2
        self.probMid = 1-self.probUp-self.probDown

        self.strike = envParameters['opt']['strike']
        self.T = envParameters['opt']['T']

        self.ttmData = self._ttmFull(self.n_episodes, self.n_samples, self.dt)[0,:]
        self.ttTransitions = self._createData()
        self.cumSumTransitions = np.cumsum(self.ttTransitions,axis=1).astype(int)
        self.trinomialDataAllEpisodes = self.s0*pow(self.upFactor,self.cumSumTransitions)
        
        self.bsPricesAllEpisodes = np.zeros((self.trinomialDataAllEpisodes.shape[0],self.trinomialDataAllEpisodes.shape[1]))
        for it_c in list(range(self.trinomialDataAllEpisodes.shape[0])):
            for it_r in list(range(self.trinomialDataAllEpisodes.shape[1])):
                self.bsPricesAllEpisodes[it_c,it_r] = self._euro_vanilla_call(self.trinomialDataAllEpisodes[it_c,it_r], self.strike, self.ttmData[it_r], 0.0, self.sigma)
           
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=np.array([0,-self.n_samples,-5000,0,0]), high=np.array([self.n_samples+1,self.n_samples,5000,2*self.s0,self.n_samples+1]))   
 

        self.episode_step = int(0)
        self.current_step = int(0)
        print('Created Env.')
    
    def _createData(self):
        ttTransitions = np.zeros((self.n_episodes,self.n_samples)) 
        for it in range(self.n_episodes):
            tmp = np.random.choice([1,0,-1], self.n_samples-1, [self.probUp,self.probMid,self.probDown])
            ttTransitions[it,1:self.n_samples] = tmp#a zero is chosen by default in the first entry of the path.
        return ttTransitions

    #Same style time to maturity array
    def _ttmFull(self, n_sim, n_samples, dt):
        tmp=np.linspace(0,(n_samples-1)*dt,n_samples)
        return n_samples*dt*np.ones((n_sim,n_samples)) -np.tile(tmp,(n_sim,1))
    
    def _euro_vanilla_call(self, S, K, T, r, sigma):
        if(abs(T-0.0)<0.0000001):
            call = max(0.0, S-K)
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            call = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
        return call

    def _euro_vanilla_delta(self, S, K, remT, r, sigma):
        if(abs(remT-0.0)<0.0000001):
            if(S-K>0):
                delta = 1.0
            else:
                delta = -1.0
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * remT) / (sigma * np.sqrt(remT))
            delta = norm.cdf(d1, 0.0, 1.0)
        return delta

    def reset(self):
        self.current_step = 0
        q = randrange(self.n_episodes)
        self.trinomialData = self.trinomialDataAllEpisodes[q,:]
        self.bsPrices = self.bsPricesAllEpisodes[q,:]
        #state 0: counter
        #state 1: number of shares held
        #state 2: riskfree account Balance
        #state 3: new stockPrice
        #state 4: time to maturity
        self.state = np.array([self.current_step,
                 0.0,
                 self.bsPrices[0],#bank account! This is part of the optimization problem. It shoud be optimized over, or chosen right a priori.
                 self.trinomialData[self.current_step],
                 self.T*365
                 ])
        
        self.episode_step += 1
        return self.state

    def stateValue(self, state):
        return state[2] + state[3] * state[1]
        # P_{t+1} = cashNew + S_{t+1}*newholding = cashOld - S_t*actOld + S_{t+1}*(actOld + holdingOld)
        # = cashOld + S_t*holdingOld + Delta S_{t+1} (actOld + holdingsOld)
        # = P_t + Delta S_{t+1} holdingsNew
        
    def step(self, action):
        self.current_step +=1
        newHoldings = (action - (self.n_actions - 1 ) / 2 ) * 2 / (self.n_actions - 1 )
        #state 0: counter
        #state 1: number of shares held
        #state 2: riskfree account Balance
        #state 3: new stockPrice
        #state 4: time to maturity
        
        #this is the new state 
        self.state[0] = self.state[0]+1
        self.state[2] = self.state[2] - self.state[3] * (- self.state[1] + newHoldings)
        self.state[1] = newHoldings
        self.state[3] = self.trinomialData[int(self.state[0])]
        self.state[4] = self.state[4]-self.dt*365
  
        
        done = False

        if(self.trueMDP):
            reward = 0.0
        else:
            tmp = 100*1/self.s0*(self.stateValue(self.state)-self.bsPrices[self.current_step])**2
            if tmp > 10:
                tmp = 10
            reward =  -tmp/5+1
        
        if(self.current_step==self.n_samples-1):
            done = True
            tmp = 100*1/self.s0*(self.stateValue(self.state)-max(0.0,self.state[3]-self.strike))**2
            if tmp > 10:
                tmp = 10
            reward =  -tmp/5+1
        return self.state, reward, done, {}

    def render(self, mode='human', close=False):
        print('Episodes: ' + str(self.episode_step))

config = DQNConfig()

config.framework("torch") 
config.rollouts(num_rollout_workers=1) 
config.training(gamma=0.99)
config.environment(OptionHedgingEnvMinimal, env_config = env_config)

trainer = DQN(config=config) 

print('Training...')
for i in range(trainTime_epochs):
    result = trainer.train()
    if not i%10:
       print('I ran ' + str(i) + ' episodes.')
       print(pretty_print(result))

print('Training Complete.')


#Evaluate model and compute the heding performance on a number of unseen paths
if(saveData):
    def euro_vanilla_delta(S, K, remT, r, sigma):
        if(abs(remT-0.0)<0.0000001):
            if(S-K>0):
                delta = 1.0
            else:
                delta = -1.0
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * remT) / (sigma * np.sqrt(remT))
            delta = norm.cdf(d1, 0.0, 1.0)
        return delta

    def ttmFull(n_sim, n_samples, dt):
        tmp=np.linspace(0,(n_samples-1)*dt,n_samples)
        return n_samples*dt*np.ones((n_sim,n_samples)) -np.tile(tmp,(n_sim,1))
    
    ttmData = ttmFull(1000, env_config['globalEnv']['n_samples'], env_config['gbm']['dt'])

    #compute the trained model's hedges
    print('Computing model hedges...')
    env = OptionHedgingEnvMinimal(env_config)
    testDataAllEpisodes = np.zeros((1000,env_config['globalEnv']['n_samples']-1))
    modelTestDeltasEpisodes = np.zeros((1000,env_config['globalEnv']['n_samples']-1))
    bsTestDeltasEpisodes = np.zeros((1000,env_config['globalEnv']['n_samples']-1))
    bsmValuesHedges = np.zeros((bsTestDeltasEpisodes.shape[0],1))
    dqnValuesHedges = np.zeros((bsTestDeltasEpisodes.shape[0],1))
    bsm_starting_price = 0.0

    print(testDataAllEpisodes.shape[0])
    for it_c in range(0, testDataAllEpisodes.shape[0]):
        done = False
        obs = env.reset()
        episode_reward = 0
        it_r = int(0)
        bsmValuesHedges[it_c] = env.bsPrices[0]
        dqnValuesHedges[it_c] = env.bsPrices[0]
        while not done:
            testDataAllEpisodes[it_c,it_r] = obs[3]
            bsTestDeltasEpisodes[it_c,it_r] = euro_vanilla_delta(obs[3], env.strike, env.ttmData[it_r], 0.0, env.sigma)
            action = trainer.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            modelTestDeltasEpisodes[it_c,it_r] = env.state[1]
            it_r += 1

    print('Evaluating model hedges...')
    for count in range(0, testDataAllEpisodes.shape[1]-1):
        bsmValuesHedges = bsmValuesHedges + ((testDataAllEpisodes[:,count + 1 : count + 2]) - (testDataAllEpisodes[:,count : count + 1])) * bsTestDeltasEpisodes[:,count: count + 1]
        dqnValuesHedges = dqnValuesHedges + ((testDataAllEpisodes[:,count + 1 : count + 2]) - (testDataAllEpisodes[:,count : count + 1])) * modelTestDeltasEpisodes[:,count : count+1]
    strikeT = np.ones(dqnValuesHedges.shape)* env_config['opt']['strike']
    valuesOptions = - 0.5* (((testDataAllEpisodes[:,-1]).reshape(-1,1) - strikeT) + np.abs((testDataAllEpisodes[:,-1]).reshape(-1,1) - strikeT))

    print(bsTestDeltasEpisodes[0:2,])
    print(modelTestDeltasEpisodes[0:2,])
    print('Saving output...')
    np.save('results/gbmTestDataAllEpisodes_' + str(env_config['globalEnv']['trueMDP']) + '_' + str(saveId) + '.npy', testDataAllEpisodes)
    np.save('results/valuesOptions_' + str(env_config['globalEnv']['trueMDP']) + '_' + str(saveId) + '.npy', valuesOptions)
    np.save('results/dqnValuesHedges_' + str(env_config['globalEnv']['trueMDP']) + '_' + str(saveId) + '.npy', dqnValuesHedges)
    np.save('results/bsmValuesHedges_' + str(env_config['globalEnv']['trueMDP'])  + '_' + str(saveId) + '.npy', bsmValuesHedges)
    np.save('results/bsTestDeltasEpisodes_' + str(env_config['globalEnv']['trueMDP']) + '_' + str(saveId) + '.npy', bsTestDeltasEpisodes)
    np.save('results/modelTestDeltasEpisodes_' + str(env_config['globalEnv']['trueMDP'])  + '_' + str(saveId) + '.npy', modelTestDeltasEpisodes)