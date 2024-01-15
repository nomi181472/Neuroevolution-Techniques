import torch
import torch.nn as nn
import gym
import numpy as np
import random
import copy 
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import os
from torch.utils.tensorboard.writer import SummaryWriter
import asyncio


def set_seed(seed_value=42):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


env_name="LunarLander-v2"
gym_env=gym.make(env_name,continuous=True)




class NeuralNetwork(nn.Module):
    def __init__(self,input,output):
        
        super(NeuralNetwork,self,).__init__()
        #TODO generic neural network
        self.seq=nn.Sequential(
            nn.Linear(input,36,bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(36,36,bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(36,output,bias=False),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.seq(x)
        #action=torch.argmax(x).item()
        return x

class Solution:
    def __init__(self,net:NeuralNetwork,myenv,max_iters=100,device="cpu",):
        self.reward=float("-inf")
        self.rewards=[]
        self.net=net
        self.device=device
        self.net.to(self.device)
        self.max_iters=max_iters
        self.myenv=myenv
    def get_median(self):
        return np.median(self.rewards)
    def assign_random_values(self,):
        flat_weights=self.get_flatten_weights()
        index=0
        for layer in self.net.children():
            if isinstance(layer,nn.Linear):
                indicies=layer.in_features*layer.out_features
                
                num=torch.sqrt(torch.tensor(2)/torch.tensor(layer.in_features),)
                flat_weights[index:index+indicies]=flat_weights[index:index+indicies]*num
                index=index+indicies
        self.update_wights(flat_weights)
    def get_flatten_weights(self,):
        return parameters_to_vector(self.net.parameters()).to(self.device)
    
    def update_wights(self,flat_list:torch.Tensor):
        vector_to_parameters(flat_list.to(self.device),self.net.parameters())
    def save_weights(self,path):
        torch.save(self.net.to("cpu").state_dict(), f"{path}")
    def load_weights(self,path):
        self.net.load_state_dict(torch.load(path))
    def normalize_weights(self):
        self.update_wights(F.normalize(self.get_flatten_weights(), p=2, dim=0))     
    @torch.no_grad()    
    def calculate_reward(self,):

        done=False
        terminated=False
        state,_=gym_env.reset()
        self.net.to(self.device)
        total_reward=0
        states=torch.empty(0)
        for _ in range(self.max_iters+100):
            tensor_state=torch.Tensor( state)
            action=self.net(tensor_state.to(self.device))
            states=torch.cat((states, tensor_state), dim=0)
            next_state,reward,done,terminated,info=gym_env.step(np.array(action))
            state=next_state
            total_reward=total_reward+reward
            if done or terminated:
                break
        self.reward=round(total_reward,2)
        self.rewards.append(round(total_reward,2))
        return states

    async def calculate_reward_multi(self):
        with torch.no_grad():
            done=False
            terminated=False
            state,_=self.myenv.reset()
            self.net.to(self.device)
            total_reward=0
            for _ in range(self.max_iters+100):
                action=self.net(torch.Tensor( state).to(self.device))
                next_state,reward,done,terminated,info=self.myenv.step(action.numpy())
                state=next_state
                total_reward=total_reward+reward
                if done or terminated:
                    break
            self.reward=round(total_reward,2)
            self.rewards.append(round(total_reward,2))
  
    @torch.no_grad()    
    def test(self,):
        total_reward=0
        done=False
        terminated=False
        state,_=gym_env.reset()
        for _ in range(self.max_iters+1000):
            action=self.net(torch.Tensor( state).to(self.device))
            
            next_state,reward,done,terminated,info=gym_env.step(action)
            state=next_state
            total_reward=total_reward+reward
            if done or terminated:
                break

        
        self.reward=total_reward



class Cuckoo:
    def __init__(self,input:int,output:int,population_size=100,generation_number=100,beta=1.5,lb=[-5, -5],ub=[5, 5],pa=0.5,alpha=0.4):
        self.population_size=population_size
        self.input=input
        self.output=output
        self.Populations=[]
        self.overall_median=[]
        self.generation_number=generation_number
        self.highest_reward=0
        self.highest_median=0
        self.beta=beta
        self.pa=pa
        self.lb=lb
        self.ub=ub
        self.alpha=alpha
        self.writer=SummaryWriter("logs")
        self.device=("cuda" if torch.cuda.is_available() else "cpu")
        self.env_stds=torch.empty(0)
        self.env_means=torch.empty(0)
    def get_normal(self,length):
        return torch.normal(mean=torch.mean(self.env_means),std=torch.std(self.env_stds),size=(length,))

    def fit(self,):
        # Initial best solution
        self.Populations=[Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make(env_name,continuous=True),device=self.device) for _ in range(self.population_size)]
        #evaluate performance
        for sol in self.Populations:
            states=sol.calculate_reward()
            self.env_means=torch.cat((self.env_means,states),dim=0)
            self.env_stds=torch.cat((self.env_stds,states),dim=0)




        for generation in range(self.generation_number):
            t1=time.time()

            total_selecte_for_nesting=int(self.population_size*self.alpha)
            new_popuation=copy.deepcopy(self.Populations[:total_selecte_for_nesting]) 
            self.Populations=sorted(self.Populations,key=lambda x: x.reward,reverse=True)
            best_nest = copy.deepcopy(self.Populations[0]) 

            for i in range(total_selecte_for_nesting):
                # Generate new solutions (cuckoo eggs) using Levy flights
                ith_nest=new_popuation[i].get_flatten_weights()
                step = torch.tensor(self.levy_flight(),dtype=torch.float32) 
                indicies=torch.randint(1, len(ith_nest)-1, (torch.randint(1,len(ith_nest)-1,size=(1,)).item(),))
                ith_nest[indicies] = ith_nest[indicies] + step * torch.tensor(self.get_normal(len(ith_nest[indicies])),dtype=torch.float32)
                new_popuation[i].update_wights(ith_nest)
                states=new_popuation[i].calculate_reward()
                self.env_means=torch.cat((self.env_means,states),dim=0)
                self.env_stds=torch.cat((self.env_stds,states),dim=0)

                # Replace nests with better solutions
                if new_popuation[i].reward >self.Populations[i].reward:
                    self.Populations[i] =copy.deepcopy(new_popuation[i])

                    # Update best solution if found
                    if self.Populations[i].reward> best_nest.reward:
                        print(f"best solution updated from {best_nest.reward} to {self.Populations[i].reward}")
                        best_nest = copy.deepcopy(self.Populations[i])

           
            # Abandon a fraction (Pa) of worse nests
            abondon_egg_range=self.population_size-total_selecte_for_nesting

            selected_indices = np.random.choice(range(total_selecte_for_nesting-1, self.population_size-1), size=abondon_egg_range, replace=False)
            for i in selected_indices:
                new_nest = Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make(env_name,continuous=True),device=self.device)
                weights=new_nest.get_flatten_weights()
                new_weights=self.get_normal(len(weights))
                new_nest.update_wights(new_weights+best_nest.get_flatten_weights())
                states=new_nest.calculate_reward()
                self.env_means=torch.cat((self.env_means,states),dim=0)
                self.env_stds=torch.cat((self.env_stds,states),dim=0)
                if new_nest.reward>self.Populations[i].reward:
                    self.Populations[i]=copy.deepcopy(new_nest)


            self.statistics(generation=generation)

            print(f"Time Taken: {round(time.time()-t1,2)}")
        self.writer.close()
    async def fit_multithreads(self,):
        self.Populations=[Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make(env_name,continuous=True),device=self.device) for _ in range(self.population_size)]
        for i in range(self.population_size):
            self.Populations[i].assign_random_values()
        
        for generation in range(self.generation_number):
            t1=time.time()

            processes=[]
            
            for sol in self.Populations:
                task = asyncio.create_task(sol.calculate_reward_multi())
                processes.append(task)


            await asyncio.gather(*processes)


            self.statistics(generation=generation)
            self.evolve_population()
            print(f"Time Taken: {round(time.time()-t1,2)}")
        self.writer.close()
    def levy_flight(self):
        beta=self.beta
        """Generates a step size using Levy flight."""
        sigma = (np.random.gamma(1 + beta, 1) * np.sin(np.pi * beta / 2) /
                (np.pi * beta * np.power(2, (beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        return u
    def eval(self,weight_paths,num_of_times_repeat=10):
        rewards=[]
        sol=Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make(env_name,continuous=True),device=self.device)
        sol.load_weights(weight_paths)
        rewards=[]
        for i in range(num_of_times_repeat):
            sol.calculate_reward()
            rewards.append(sol.reward)

            print(f"Reward:{sol.reward}")
        print(f"Mean: {round(np.mean(rewards))}")
        print(f"median: {round(np.median(rewards))}")
    def statistics(self,generation):
        max_solution=self.Populations[-1]
        max_solution_by_median=self.Populations[-1]
        
        reward_of_each_sol=[]
        min_reward=float(1000)

        for solution in self.Populations:
            #max
            if solution.reward>max_solution.reward:
                max_solution=solution
            if min_reward>solution.reward:
                min_reward=solution.reward
            if solution.get_median()>max_solution_by_median.get_median():
                max_solution_by_median=solution
            reward_of_each_sol.append(solution.reward)
            
        med_reward=round(np.median(reward_of_each_sol))
        if max_solution.reward>self.highest_reward:
            self.highest_reward=max_solution.reward
            max_solution.save_weights(f"weights_{generation}_individual_max_cross.pth")
            max_solution_by_median.save_weights(f"weights_{generation}_coperative_max_cross.pth")
            #save with highest median

        if self.highest_reward!=0 and self.highest_reward!= max_solution.reward and max_solution.reward/self.highest_reward>=0.98:
            max_solution.save_weights(f"weights_{generation}_individual_near_highest.pth")
            max_solution_by_median.save_weights(f"weights_{generation}_coperative_near_highest.pth")
            
        if med_reward>=self.highest_median:
            max_solution.save_weights(f"weights_{generation}_individual_median_cross.pth")
            self.highest_median=med_reward
            max_solution_by_median.save_weights(f"weights_{generation}_coperative_median_cross.pth")


        print(f"Generation:{generation}/{self.generation_number} Med:{med_reward}<->MedPrev:{self.highest_median} MaxReward:{max_solution.reward} MinReward:{min_reward} RewardPrev: {self.highest_reward}  ")
        self.writer.add_scalar("Current Median",med_reward,generation)
        self.writer.add_scalar("Past Median",self.highest_median,generation)
        self.writer.add_scalar("Current MaxReward",max_solution.reward,generation)
        self.writer.add_scalar("Current MinReward",min_reward,generation)
        self.writer.add_scalar("Past MaxReward",self.highest_reward,generation)
        self.writer.add_histogram("Best Performer",np.array(max_solution.rewards),generation)

if __name__ == '__main__':
    

    TOTAL_STATES=gym_env.observation_space.shape[0]
    TOTAL_ACTIONS=gym_env.action_space.shape[0]
    print(f"TOTAL_STATES:{TOTAL_STATES}")
    print(f"TOTAL_ACTIONS:{TOTAL_ACTIONS}")
    POPULATION_SIZE = 80
    MAX_GENERATION = 2000

    MUTATION_RATE = 0.4
    CROSSOVER_RATE = 0.9
    
    set_seed(4)
    agent=Cuckoo(TOTAL_STATES,TOTAL_ACTIONS,POPULATION_SIZE,generation_number=MAX_GENERATION)
    agent.fit()
    #asyncio.run( agent.fit_multithreads())


    #agent.eval("weights_1291_individual_near_highest.pth",30)

#highest 174,167



