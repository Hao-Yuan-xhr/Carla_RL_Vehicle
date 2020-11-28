# -*- coding: utf-8 -*-
"""
Created on SAT NOV 28 14:26:21 2020

@author: Hao Yuan

Double DQN

"""

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Input, BatchNormalization, Dropout
from keras.optimizers import Adam,SGD
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend as backend
from threading import Thread
from tqdm import tqdm
import carla
import matplotlib.pyplot as plt

# GPU choose
os.environ['CUDA_VISIBLE_DEVICES']='0'


IM_WIDTH = 320
IM_HEIGHT = 240
SECONDS_PER_EPISODE = 30
EPISODES = 15

REPLAY_MEMORY_SIZE = 6000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 20
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 25
UPDATE_REWARD_EVERY = 100
MODEL_NAME = "Double DQN".format(int(time.time()))

MEMORY_FRACTION = 0.2
MIN_REWARD = 4000

DISCOUNT = 0.95
epsilon = 1
EPSILON_DECAY = 0.995 
MIN_EPSILON = 0.0001

AGGREGATE_STATS_EVERY = 10


# copy tensorboard type 
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
    
    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()
    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class CarEnv:
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.sps=self.map.get_spawn_points()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.driving_reward = 0
        self.aim_reward = 0
        self.v_reward = 0
        
    def reset(self):
        self.collision_flag = False        
        self.lane_flag = False
        self.actor_list = []
        self.i2 = np.zeros(shape=(self.im_height, self.im_width))  
        self.distance_s = 0
        
        # vehicle intial
        self.transform = carla.Transform(carla.Location(x=12, y=-1.96, z=1.32),carla.Rotation(yaw=180))
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(3)
        
        # sensors fix
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", "{}".format(self.im_width))
        self.rgb_cam.set_attribute("image_size_y", "{}".format(self.im_height))
        self.rgb_cam.set_attribute("fov", "110")
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
                
        lansensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lansensor = self.world.spawn_actor(lansensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lansensor)
        self.lansensor.listen(lambda event: self.laneinvasion_data(event))
        
        # figure trigger
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_flag = True

    def laneinvasion_data(self, event):
        info_lane = set(x.type for x in event.crossed_lane_markings)
        text = [str(x).split()[-1] for x in info_lane]
        if text[0] == 'Broken':
            self.lane_flag = True
        else:
            self.lane_flag = False
        
    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        cropped = i3[116:240,:]
        i3 = np.array(cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)).reshape(124, 320, 1)
#        print(np.array(i3).shape)
        self.front_camera = i3
            
    def step(self, action):
        self.aim_reward = 0
        self.driving_reward = 0
        self.v_reward = 0
        # another sensor mechanism to find the relationship between the vehicle and different lane type
        self.waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=False)
        
        # action define
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=-0.3*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.3*self.STEER_AMT))
        
        # distance from the current position to the start point (x=12, y=-1.96) 
        distance_r = int(math.sqrt((self.vehicle.get_location().x - 12)**2 + (self.vehicle.get_location().y + 1.96)**2))
        # distance from the current position to the end point (here we hope the vehicle to across the line  y = 40)
        distance_h = int(abs(self.vehicle.get_location().y - 40))
     
        # one frame reward desgin
        # v_reward 
        v = self.vehicle.get_velocity()
        self.v_reward = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # driving_reward
        if not self.waypoint:
            done = True
            self.driving_reward = -100
            
        elif self.collision_flag:
            done = True
            self.driving_reward = -100
        
        elif distance_h < 1:
            done = True
            self.driving_reward = 1000
               
        elif self.lane_flag:
            done = False # allow to across the lane
            self.driving_reward = -3
            self.lane_flag = False        
        else:
             done = False
             self.driving_reward = 0.5

        # aim_reward 
        if distance_r != self.distance_s:
            self.aim_reward = 2 * (distance_r - self.distance_s) 
            self.distance_s = distance_r
        
        # sum    
        reward = self.driving_reward + self.aim_reward + (self.v_reward - 1)
       
        #time limit
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

class NatureDQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}".format(MODEL_NAME))
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        model = Sequential()
        input_1 = Input(shape = (124, 320, 1))
        conv1 = Conv2D(32,(8,8), activation='relu', padding='same')(input_1)
        pool_1 = MaxPooling2D(pool_size=(5, 5),strides=(4, 4),padding='same')(conv1)
        conv2 = Conv2D(64,(4,4), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding='same')(conv2)
        conv3 = Conv2D(64,(3,3), activation='relu', padding='same')(pool_2)
        pool_3 = MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding='same')(conv3)
        flatten_1 = Flatten()(pool_3)
        drop_1 = Dropout(0.25)(flatten_1)
        out_1 = Dense(3,activation='linear')(drop_1)
        model = Model(inputs=input_1, outputs = out_1)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    # current Q-net train
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255    
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        
        next_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            future_pre_qs_list = self.model.predict(next_states, PREDICTION_BATCH_SIZE)
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(next_states, PREDICTION_BATCH_SIZE)
            
        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                future_q = np.argmax(future_pre_qs_list[index])
                max_future_q = future_qs_list[future_q]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step
            
        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        #target copy
        if log_this_step:
            self.target_update_counter += 1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):        
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, 124, 320, 1)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 9
    ep_rewards = []
    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    config.gpu_options.allow_growth = True
    backend.set_session(tf.Session(config=config))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = NatureDQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((124, 320, 1)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()
        
        while True:
            if np.random.random() > epsilon:
                action_list = agent.get_qs(current_state)
                action = np.argmax(action_list)
                #tensorboard recode
                agent.tensorboard.update_stats(ac_0=action_list[0], ac_1=action_list[1], ac_2=action_list[2])
                #depend on your device computing
                time.sleep(1/FPS - 0.005)                
            else:                                     
                action = np.random.randint(0, 3)                
                time.sleep(1/FPS)   
            
            new_state, reward, done, _ = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            
            step += 1

            if done:
                break
        
        print(episode_reward)        

        for actor in env.actor_list:
            actor.destroy()
        
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, step=step)
            if min_reward >= MIN_REWARD:
                agent.model.save('models/{}'.format(int(time.time())))

        
        #policy for exploration        
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        
    agent.terminate = True
    trainer_thread.join()
    # agent.model.save('models/{}'.format(MODEL_NAME))
