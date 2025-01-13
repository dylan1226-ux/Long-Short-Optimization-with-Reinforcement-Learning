# Long/Short Optimization with Reinforcement Learning 
### 1. Introduction
Transforming raw stock data into predictive signals, or alphas, is commonly practiced by professionals to identify potential trading opportunities in financial markets. 
I developed and refined a reinforcement learning-based model with the optimization of a long/short strategy. 
Compared to commonly used genetic programming, the RL model directly uses the performance of a pool of alphas to optimize an alpha generator, 
taking into account the synergy involved in stock selection.

### 2. Alpha Mining Pipeline
2.1 Alpha Generator

The alpha generator uses tokens as representations and outputs a token sequence. 
Each token represents a distinct element in an alpha expression, such as a feature or an arbitrary number. 
The process is modeled as a Markov Decision Process (MDP): 
1.	State: The current token sequence that has been generated.
2.	Action Space: The set of all possible tokens that can be taken for the next step. Invalid actions will be masked in the action space.
3.	Action Mask: A filter mechanism to mask invalid actions based on the current state.
4.	Action: A token from the action space that will be appended to the token sequence.
5.	Reward: The outcome of a valid expression in the combination model. 

Based on this MDP, the generator employs the Maskable PPO algorithm in reinforcement learning to select actions.  
The model takes the state as input and outputs a distribution of actions, 
in which the actual action is sampled and appended to the token sequence.

2.2 Combination Model

When a token sequence is labeled valid from the alpha generator, 
it will be parsed into a mathematical formula and entered into the combination model. The input of the combination is a union of the new expression and the existing pool, along with their weights. The performance of input is evaluated based on the Sharpe ratio (risk-adjusted return) using a long/short strategy, which goes long and short on stocks based on alpha values. 

Gradient Descent is subsequently performed to optimize weights. 
I design the loss function as the negative of Sharpe, so minimizing the loss function effectively maximizes the Sharpe ratio. 
An optimized Sharpe ratio is eventually passed back as the reward signal to the alpha generator, encouraging the creation of high-performance alphas.

### 3. Data Structure


The **alphagen** folder contains the combination model and the reinforcement learning model.

The **alphagen_qlib** folder handles the long/short strategy and data processing.

The **alphavalue_gen.py** script acts as a parser for generated expressions, converting them into numerical values and returning a dataframe of alpha values. 
For example, given an expression like `Add($close, $high)`, the parser will process this expression and output the corresponding dataframe.

To run the code, execute **my_traning.py**, specifying the checkpoint and log paths as needed.

### 4, Long/Short Strategy
The Long/Short strategy takes long positions in stocks with high alpha values and short positions in those with low alpha values. 
For initialization, a dummy dataframe is used to filter out untradable stocks, assigning them a null value of 0. 
The return rate is calculated based on the closing prices, depending on the holding period for the stocks. 

The input alpha values are ranked and normalized row-wise, then assigned a score. 
Long and short positions are determined based on these scores. 
Finally, the strategy's performance is evaluated using the Sharpe ratio, which provides a measure of risk-adjusted return.

**This strategy is recalculated each time the alpha pool adds a new expression, 
as well as during gradient descent to optimize the weights for the highest Sharpe ratio, 
which serves as the reward signal for the alpha generator.**

### Citation
Yu, Shuo, Xue, Hongyan, Ao, Xiang, Pan, Feiyang, He, Jia, Tu, Dandan, He, Qing. (2023). 
"Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning." In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. DOI: [10.1145/3580305.3599831](https://doi.org/10.1145/3580305.3599831)
