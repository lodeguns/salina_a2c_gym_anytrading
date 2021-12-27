# Facebook Salina - Gym_AnyTrading
Slight modification of Facebook Salina Reinforcement Learning - [A2C GPU example](https://github.com/facebookresearch/salina/tree/main/salina_examples/rl/a2c) for financial series.

The gym FOREX data are provided by gym_anytrading library [GitHub Page](https://github.com/AminHP/gym-anytrading)

With respect to the traditional CartPole-V0 gym, the _stock_func_ is designed to provide in input a
FOREX trading gym:

```
def stock_func(max_episode_steps,seed=123, window_size=10, size_sample=100):
    df =  FOREX_EURUSD_1H_ASK.copy()
    start_index = window_size
    if size_sample < 0:              ### put size_sample=-1 to consider the whole dataset.
        end_index = len(df)
    else:
        end_index = size_sample
    env = TimeLimit(gym.make('forex-v0', df=df, window_size=window_size, frame_bound=(start_index, end_index)), max_episode_steps=max_episode_steps)
    env.seed(seed)
    return env
```

A double tensor with diff close and relative gains are given in the _gen_state_ function.
The latter transforms the 'env/obs' tensors collected from AutoResetGymAgent 
into suitable tensors for Policy/Critic agent neural networks.


```
def _gen_state(observation):
    index = torch.tensor([0])
    diff_close = torch.transpose(torch.index_select(observation[0], 1, index), 1, 0)
    index2 = torch.tensor([1])
    buy_sell = torch.transpose(torch.index_select(observation[0], 1, index2), 1, 0)
    observation = torch.squeeze(torch.stack([diff_close, buy_sell], dim=1))
    return  observation
```






I take no responsibility for the use of the code.
It is a simple test of SALINA's potential for financial problems.

 


The software license remains the one indicated in the source code and respectively linked to the official Facebook SALINA repository [GitHub Page](https://github.com/facebookresearch/salina)


Francesco Bardozzo, PhD
fbardozzo@unisa.it - NeuroneLab - University of Salerno - Italy

Pravesh Kriplani, PhD
pkriplani@unisa.it - NeuroneLab - University of Salerno - Italy 


