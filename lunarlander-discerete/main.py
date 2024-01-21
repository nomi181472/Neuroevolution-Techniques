import torch
import torch.nn as nn
import gym
import numpy as np
import random
import copy 
import time
import matplotlib.pyplot as plt

from torch.nn.utils import parameters_to_vector, vector_to_parameters
import os
from torch.utils.tensorboard.writer import SummaryWriter
import asyncio


env_name="LunarLander-v2"
gym_env=gym.make(env_name)

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
            nn.Linear(input,24,bias=True),
            nn.LeakyReLU(-0.1),
            nn.Linear(24,24,bias=True),
            nn.LeakyReLU(-0.1),
            nn.Linear(24,output,bias=True),
           nn.Softmax(dim=0)
        )
    def forward(self,x):
        x=self.seq(x)
        x=torch.argmax(x).item()
        return x
class Solution:
    def __init__(self,net:NeuralNetwork,myenv,max_iters=100,device="cpu",
                 buffer_size=100):
        self.rewards=np.empty(0)
        self.buffer_size=buffer_size
        self.net=net
        self.device=device
        self.net.to(self.device)
        self.max_iters=max_iters
        self.myenv=myenv
    def get_median(self):
        return np.median(self.rewards)
    def get_fitness_value(self,max_score=280,length_weight=0.7,personal_reward_weight=0.3,):
        score=np.median(self.rewards)+ length_weight*len(self.rewards)+self.rewards[-1]*personal_reward_weight
        score=score+np.sum(self.rewards>max_score)
        return score
    def get_flatten_weights(self,):
        return parameters_to_vector(self.net.parameters()).to(self.device)
    def update_wights(self,flat_list:torch.Tensor):
        vector_to_parameters(flat_list.to(self.device),self.net.parameters())
    def save_weights(self,path):
        torch.save(self.net.to("cpu").state_dict(), f"{path}")
    def load_weights(self,path):
        self.net.load_state_dict(torch.load(path))    
    @torch.no_grad()    
    def calculate_reward(self,):

        done=False
        terminated=False
        state,_=gym_env.reset()
        self.net.to(self.device)
        total_reward=0
        for _ in range(self.max_iters+100):
            action=self.net(torch.Tensor( state).to(self.device))
            next_state,curr_step_reward,done,terminated,info=gym_env.step(np.array(action))
            state=next_state
            total_reward=total_reward+curr_step_reward
            if done or terminated:
                break
        self.update_reward_buffer(total_reward)
    @torch.no_grad()  
    async def calculate_reward_async(self):
        with torch.no_grad():
            done=False
            terminated=False
            state,_=self.myenv.reset()
            self.net.to(self.device)
            total_reward=0
            for _ in range(self.max_iters+100):
                action=self.net(torch.Tensor( state).to(self.device))
                next_state,curr_step_reward,done,terminated,info=self.myenv.step(action)
                state=next_state
                total_reward=total_reward+curr_step_reward
                if done or terminated:
                    break
            self.update_reward_buffer(total_reward)

    def update_reward_buffer(self, total_reward):
        if len(self.rewards)+1>self.buffer_size:
            self.rewards=np.delete(self.rewards,-1)
        self.rewards=np.append(self.rewards,round(total_reward,2))

  
    @torch.no_grad()    
    def test(self,):
        total_reward=0
        done=False
        terminated=False
        state,_=self.myenv.reset()
        for _ in range(self.max_iters+1000):
            action=self.net(torch.Tensor( state).to(self.device))
            
            next_state,curr_step_reward,done,terminated,info=self.myenv.step(action)
            state=next_state
            total_reward=total_reward+curr_step_reward
            if done or terminated:
                break
        self.rewards=np.append(self.rewards,total_reward)



class GA:
    def __init__(self,input:int,output:int,population_size=100,generation_number=100,threshold_rewards=280):
        self.population_size=population_size
        self.input=input
        self.output=output
        self.Populations=[]
        self.overall_median=[]
        self.generation_number=generation_number
        self.highest_reward=0
        self.highest_median=0
        self.last_collective_performer=-1
        self.threshold_rewards=threshold_rewards
        self.writer=SummaryWriter("logs")
        self.device=("cuda" if torch.cuda.is_available() else "cpu")
    def fit(self,):
        self.Populations=[Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make(env_name,),device=self.device) for _ in range(self.population_size)]

        
        for generation in range(self.generation_number):
            t1=time.time()
            for sol in self.Populations:
                sol.calculate_reward()

            self.statistics(generation=generation)
            self.evolve_population()
            print(f"Time Taken: {round(time.time()-t1,2)}")
        self.writer.close()
    async def fit_async(self,):
        self.Populations=[Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym.make(env_name),device=self.device) for _ in range(self.population_size)]
        
        for generation in range(self.generation_number):
            t1=time.time()

            processes=[]
            
            for sol in self.Populations:
                task = asyncio.create_task(sol.calculate_reward_async())
                processes.append(task)


            await asyncio.gather(*processes)


            self.statistics(generation=generation)
            self.evolve_population()
            print(f"Time Taken: {round(time.time()-t1,2)}")
        self.writer.close()

    def eval(self,weight_paths,num_of_times_repeat=10):
        sol=Solution(NeuralNetwork(input=self.input,output=self.output),myenv=gym_env)
        sol.load_weights(weight_paths)
        rewards=[]
        for i in range(num_of_times_repeat):
            sol.calculate_reward()
            rewards.append(sol.rewards[-1])
        print(f"Mean: {round(np.mean(rewards))}")
        print(f"median: {round(np.median(rewards))}")
        print(f"Rewards:{sorted(rewards,reverse=True)}")
    def statistics(self,generation):
        max_solution=self.Populations[-1]
        max_solution_by_median=self.Populations[-1]
        reward_of_each_sol=[]
        min_reward=float(1000)

        for solution in self.Populations:
            #max
            if solution.rewards[-1]>max_solution.rewards[-1]:
                max_solution=solution
            if min_reward>solution.rewards[-1]:
                min_reward=solution.rewards[-1]
            if solution.get_fitness_value(max_score=280)>max_solution_by_median.get_fitness_value(max_score=280):
                max_solution_by_median=solution
            reward_of_each_sol.append(solution.rewards[-1])

        best_individual_reward=max_solution.rewards[-1]
        med_reward=round(np.median(reward_of_each_sol))
        best_collective=sorted(self.Populations,key=lambda x: sum(x.rewards>self.threshold_rewards),reverse=True)[0]

        #save with highest reward
        if best_individual_reward>self.highest_reward:
            self.highest_reward=best_individual_reward
            max_solution.save_weights(f"best_individual.pth")
        #save weights with the height threshold
        total_good_performer=sum(best_collective.rewards>self.threshold_rewards)
        if sum(best_collective.rewards>self.threshold_rewards)>0 and  total_good_performer>=self.last_collective_performer:
            print(f"best collective saved {total_good_performer}")
            self.last_collective_performer=total_good_performer
            best_collective.save_weights(f"best_collective.pth")    
        
        #save with highest median reward
        if med_reward>=self.highest_median:
            self.highest_median=med_reward
            max_solution_by_median.save_weights(f"best_individual_median.pth")

        print(f"Generation:{generation}/{self.generation_number},history_highest_reward:{self.highest_reward} ")
        print(f"best gene has total experience of {len(max_solution_by_median.rewards)} generations,median:{round(max_solution_by_median.get_median(),2)} score:{round(max_solution_by_median.get_fitness_value(max_score=self.threshold_rewards))} last 4 rewards: {max_solution_by_median.rewards[-4:]}")
        self.writer.add_scalar("current population median reward",med_reward,generation)
        self.writer.add_scalar("history population highest median reward",self.highest_median,generation)
        self.writer.add_scalar("current highest reward",best_individual_reward,generation)
        self.writer.add_scalar("current min reward",min_reward,generation)
        self.writer.add_scalar("history highest reward",self.highest_reward,generation)
    def selection(self,):
        parents=[]
        rand_num_for_pop_parents=np.random.randint(4,self.population_size)
        self.Populations= sorted(self.Populations,key=lambda x: x.get_fitness_value(max_score=self.threshold_rewards),reverse=True)
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
            child_sol= child_sol=Solution(NeuralNetwork(self.input,self.output),myenv=gym.make(env_name),device=self.device)
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
            child_sol=Solution(NeuralNetwork(self.input,self.output),myenv=gym.make(env_name),device=self.device)
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
            child_sol=Solution(NeuralNetwork(self.input,self.output),myenv=gym.make(env_name),device=self.device)
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
POPULATION_SIZE = 100
MAX_GENERATION = 4000

set_seed(4)
agent=GA(TOTAL_STATES,TOTAL_ACTIONS,POPULATION_SIZE,generation_number=MAX_GENERATION)
#asyncio.run(agent.fit_multithreads())
agent.fit()

agent.eval("lunarlander-discerete/best_collective (2).pth",30)

#highest 174,167




#TODO cv2 show rewards 
#TODO try to make a package
#TODO match with existing algorithm
#TODO use pso
#TODO scalar.normalize each state
#TODO Use Neat
#TODO use Deque and increase length weight