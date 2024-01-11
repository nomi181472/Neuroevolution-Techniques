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



gym_env=gym.make("LunarLander-v2")

def set_seed(seed_value=42):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)



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
           nn.Softmax(dim=0)
        )
    def forward(self,x):
        x=self.seq(x)
        x=torch.argmax(x).item()
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
        for _ in range(self.max_iters+100):
            action=self.net(torch.Tensor( state).to(self.device))
            next_state,reward,done,terminated,info=gym_env.step(np.array(action))
            state=next_state
            total_reward=total_reward+reward
            if done or terminated:
                break
        self.reward=round(total_reward,2)
        self.rewards.append(round(total_reward,2))
    @torch.no_grad()  
    async def calculate_reward_multi(self):
        with torch.no_grad():
            done=False
            terminated=False
            state=self.myenv.reset()
            self.net.to(self.device)
            total_reward=0
            for _ in range(self.max_iters+100):
                action=self.net(torch.Tensor( state).to(self.device))
                next_state,reward,done,info=self.myenv.step(action)
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



class GA:
    def __init__(self,input:int,output:int,population_size=100,crossover_rate=0.8,mutation_bit_rate_change=20,generation_number=100):
        self.population_size=population_size
        self.input=input
        self.output=output
        self.Populations=[]
        self.overall_median=[]
        self.crossover_rate=crossover_rate
        self.mutation_bit_rate_change=mutation_bit_rate_change
        self.generation_number=generation_number
        self.highest_reward=0
        self.highest_median=0
        self.writer=SummaryWriter("logs")
        self.device=("cuda" if torch.cuda.is_available() else "cpu")
    def fit(self,):
        self.Populations=[Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make("LunarLander-v2",device=self.device)) for _ in range(self.population_size)]
        for i in range(self.population_size):
            self.Populations[i].assign_random_values()
        
        for generation in range(self.generation_number):
            t1=time.time()
            for sol in self.Populations:
                #sol.normalize_weights()
                sol.calculate_reward()

            self.statistics(generation=generation)
            self.evolve_population()
            print(f"Time Taken: {round(time.time()-t1,2)}")
        self.writer.close()
    async def fit_multithreads(self,):
        self.Populations=[Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make("LunarLander-v2"),device=self.device) for _ in range(self.population_size)]
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

    def eval(self,weight_paths,num_of_times_repeat=10):
        rewards=[]
        sol=Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym_env)
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

    def selection(self,):
        parents=[]
        rand_num_for_pop_parents=np.random.randint(4,self.population_size)
        self.Populations= sorted(self.Populations,key=lambda x: x.get_median(),reverse=True)
        parents.extend(copy.deepcopy(self.Populations[:rand_num_for_pop_parents]))
        del self.Populations[:]
        return parents
    def mutate(self,child):
        total_changes = random.randint(1, len(child)-1)
        # changes_destinations = random.randint(0, int(len(child)*0.1))
        # if changes_destinations <= int(len(child)*0.1*0.5):
        for i in range(total_changes):
            limit = random.randint(0, len(child)-1)
            mutation = random.randint(-1,1 )
            child[ limit] =child[ limit] + mutation
        return child

    def even_odd_crossover(self,selected_parents):
        
        new_population = []
        total_selected_parents_for_mating=len(selected_parents)
        for i in range(self.population_size-total_selected_parents_for_mating):
            child = []
            n1 = random.randint(0, total_selected_parents_for_mating-1)
            parent_1 = selected_parents[n1].get_flatten_weights()
            # del parents[n]
            n2 = random.randint(0, total_selected_parents_for_mating-1)
            while n2 == n1:
                n2 = random.randint(0, total_selected_parents_for_mating-1)
            parent_2 = selected_parents[n2].get_flatten_weights()
            for i in range(len(parent_1)):
                if i % 2 == 0:
                    child.append( parent_1[ i])
                else:
                    child.append( parent_2[ i])
            mutated_child = self.mutate(child)
            child_sol= child_sol=Solution(NeuralNetwork(self.input,self.output),myenv=gym.make("LunarLander-v2"),device=self.device)
            child_sol.update_wights(torch.tensor(mutated_child,dtype=torch.float32,device=self.device))
            new_population.append(child_sol)
        self.Populations.extend(new_population)
        self.Populations.extend(selected_parents)
    def one_point_crossover(self,selected_parents):
        
        new_population = []
        total_selected_parents_for_mating=len(selected_parents)
        for i in range(self.population_size-total_selected_parents_for_mating):
            child = []
            n1 = random.randint(0, total_selected_parents_for_mating-1)
            parent_1 = selected_parents[n1].get_flatten_weights()
            # del parents[n]
            n2 = random.randint(0, total_selected_parents_for_mating-1)
            while n2 == n1:
                n2 = random.randint(0, total_selected_parents_for_mating-1)
            parent_2 = selected_parents[n2].get_flatten_weights()
            rand_index_Point=torch.randint(0,10,size=(1,)).item()
            child.extend(parent_1[:rand_index_Point])
            child.extend(parent_2[rand_index_Point:])
 
            mutated_child = self.mutate(child)
            child_sol=Solution(NeuralNetwork(self.input,self.output),myenv=gym.make("LunarLander-v2"),device=self.device)
            child_sol.update_wights(torch.tensor(mutated_child,dtype=torch.float32,device=self.device))
            new_population.append(child_sol)
        self.Populations.extend(new_population)
        self.Populations.extend(selected_parents)
    
    def average_crossover(self,selected_parents):
        
        new_population = []
        total_selected_parents_for_mating=len(selected_parents)
        for i in range(self.population_size-total_selected_parents_for_mating):
            child = []
            n1 = random.randint(0, total_selected_parents_for_mating-1)
            parent_1 = selected_parents[n1].get_flatten_weights()
            # del parents[n]
            n2 = random.randint(0, total_selected_parents_for_mating-1)
            while n2 == n1:
                n2 = random.randint(0, total_selected_parents_for_mating-1)
            parent_2 = selected_parents[n2].get_flatten_weights()
            child.extend( (parent_1+parent_2)/2)
 
            mutated_child = self.mutate(child)
            child_sol=Solution(NeuralNetwork(self.input,self.output),myenv=gym.make("LunarLander-v2"),device=self.device)
            child_sol.update_wights(torch.tensor(mutated_child,dtype=torch.float32,device=self.device))
            new_population.append(child_sol)
        self.Populations.extend(new_population)
        self.Populations.extend(selected_parents)
    
    
    def evolve_population(self,):
        
        
        # Perform crossover
        selected_parents = self.selection()
        
        self.one_point_crossover(selected_parents)



TOTAL_STATES=gym_env.observation_space.shape[0]
TOTAL_ACTIONS=gym_env.action_space.n
POPULATION_SIZE = 40
MAX_GENERATION = 2000

MUTATION_RATE = 0.4
CROSSOVER_RATE = 0.9

set_seed(4)
agent=GA(TOTAL_STATES,TOTAL_ACTIONS,POPULATION_SIZE,generation_number=MAX_GENERATION)
#asyncio.run(agent.fit_multithreads())


agent.eval("lunarlander-discerete\weights_1956_individual_near_highest.pth",30)

#highest 174,167




#TODO cv2 show rewards 
#TODO try to make a package
#TODO try to use multibatchprocessing
#TODO match with existing algorithm
#TODO Research on GA how to select, crossover and mutation
#TODO save weights on both median and max reward
#TODO comparies boths