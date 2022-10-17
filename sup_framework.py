import keras.backend as K
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from PointCloud_Generator.pc_generator_coverage import PcdGen_Cov
from RL_Framework.utils.env_utils import action_mapping, get_cam_pose_and_rotation, get_camera_transform_from_euler
from RL_Framework.utils.env_utils import get_output_dim
from RL_Framework.utils.pointcloud_utils import vis_mesh_and_cam_pos
import open3d as o3d
from RL_Framework.Models.keras.Pointnet_NN_Model import NN_Model
from RL_Framework.Models.keras.Pointnet_NN_Model_batch_norm import NN_Model_2
from RL_Framework.Models.keras.Pointnet_2 import get_pointnet_model, OrthogonalRegularizer
from RL_Framework.Models.keras.Pointnet_2_ohne_ba_norm import get_pointnet_model_2
from RL_Framework.Models.keras.Pointnet_NN_Model_split_out import NN_Model_split_out
from RL_Framework.Models.utils import StepDecay
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from scipy.spatial.transform import Rotation
import numpy as np
import os
from math import pi
from random import choice
import pickle


class Sup_Framework():
    """Framework that includes all functions to train and evaluate a keras network
    to predict the camera translation and rotation given a encoding matrix
    """
    def __init__(self, action_type, epochs, translation=True):
        self.resolution = [960, 600]
        self.fov_degree = [27, 25]
        self.binary_encoding = True
        self.cam_workspace = [30.0, 50.0]
        self.downsampling_factor = 0.01
        self.file_type = "stl"
        self.dim = 2048
        self.sensor_debug = False
        self.proj_bool = False
        self.proj_trans = [0,5,0]
        self.cam_workspace_bool=True
        self.CamRadius = 50
        
        self.scanner = PcdGen_Cov(
            downsampling_factor=self.downsampling_factor,
            file_type=self.file_type,
            dim=self.dim,
            scanner_config={
                "resolution": [960, 600],
                "fov_degree": [27, 25],
                "cam_workspace": [30.0, 50.0],
                "cam_workspace_bool": True,
                "sensor_debug": False,
                "proj_bool": False,
                "trans_proj": [0.0, 5.0, 0.0]
            }
        )

        self.translation = translation
        
        BASE_PATH = os.path.abspath(os.getcwd())
        data_dir = os.path.join(BASE_PATH, 'Data', 'abc')
        mesh_dir = os.path.join(data_dir, self.file_type)
        # names = [n for n in os.listdir(mesh_dir) if n.endswith(self.file_type)]
        # name = choice(names)
        name = str(32).zfill(8)
        name = name + ".stl"

        # setup scanner
        self.mesh_path = os.path.join(mesh_dir, name)
        
        self.action_type = action_type
        self.epochs = epochs

    def create_network(self, net_type:str) -> keras.Model:
        """creates keras network

        Args:
            net_type (str): name of network, possible net_type are listed below

        Returns:
            keras.Model:
        """
        if net_type == "Pointnet_2":
            self.model = get_pointnet_model(get_output_dim(self.action_type), 2048, 4)
        elif net_type == "Pointnet_NN_Model":
            if self.translation:
                self.model = NN_Model(get_output_dim(self.action_type))
                #self.model = NN_Model(16)

            else:
                self.model = NN_Model(2)
        elif net_type == "Pointnet_2_ohne_ba_norm":
            self.model = get_pointnet_model_2(get_output_dim(self.action_type), 2048, 4)
        elif net_type == "Pointnet_NN_Model_batch_norm":
            self.model = NN_Model_2(get_output_dim(self.action_type))
        elif net_type == "Pointnet_NN_Model_split_out":
            self.model = NN_Model_split_out(get_output_dim(self.action_type))
        #print(self.model.summary())

    def create_dataset(self):
        """function to create dataset"""
        dataset_length = 200 #how long the dataset is supposed to be
        points_per_scan = 50 #how many points a scan must see from the object to count as valid cam position for the dataset
        action = self.action_type #action type

        print("action_type: " + str(action) + " --- " + "dataset_length: " + str(dataset_length) + " --- " + "points_per_scan: " + str(points_per_scan))

        output_dim = get_output_dim(self.action_type)
        x_raw = np.random.uniform(-1, 1, (500000, output_dim))
        action = []
        state = []
        x = []
        init = False
        i = 0
        counter = 0
        k = 0
        while counter < dataset_length:
            print(i, end="\r")
            action_mapped = action_mapping(self.action_type, x_raw[i])
            action.append(action_mapped)
            pos, alpha, beta, gamma = get_cam_pose_and_rotation(action_type=self.action_type, actions=action_mapped, radius=self.CamRadius, initial=init)
            transform = get_camera_transform_from_euler(pos, alpha, beta, gamma, degrees=False)
            self.scanner.reset()
            self.scanner.setup(self.mesh_path)
            self.scanner.create_encoding_array()
            self.scanner.single_scan(translation=pos, alpha=alpha, beta=beta, gamma=gamma)
            if len(self.scanner.current_pcd.points) == 0:
                 i += 1
                 continue
            self.scanner.update_binary_state_array()
            array = np.asarray(self.scanner.array)

            if np.sum(array[:,3]) > points_per_scan:
                state.append(self.scanner.array)
                x.append(x_raw[i])
                counter += 1
                print("Setlength: " + str(len(state)))
                
                #batching of data
                if counter % 50 == 0:
                    y = state
                    BASE_PATH = os.path.abspath(os.getcwd())

                    with open(f"{BASE_PATH}/Data/supervised/{self.action_type}/sup_x_batch_{k}.pickle", "wb") as output:
                        pickle.dump(x, output)
                    with open(f"{BASE_PATH}/Data/supervised/{self.action_type}/sup_y_batch_{k}.pickle", "wb") as output:
                        pickle.dump(y, output)
                    k += 1
                    state = []
                    x = []
            i += 1


    def load_dataset(self, scaling=False):
        """loads a dataset

        Args:
            scaling (bool, optional): If scaling should be used. Defaults to False.
        """
        BASE_PATH = os.path.abspath(os.getcwd())
        l = len(os.listdir(f"{BASE_PATH}/Data/supervised/{self.action_type}/"))
        #loads data from multiple batched data files
        for j in range(0,int(l/2)):
            if self.translation:
                with open(f"{BASE_PATH}/Data/supervised/{self.action_type}/sup_x_batch_{j}.pickle", "rb") as data:
                    y_list = pickle.load(data)
                with open(f"{BASE_PATH}/Data/supervised/{self.action_type}/sup_y_batch_{j}.pickle", "rb") as data:
                    x_list = pickle.load(data)
            else:
                with open(f"{BASE_PATH}/Data/supervised/{self.action_type}_0T/sup_x_batch_{j}.pickle", "rb") as data:
                    y_list = pickle.load(data)
                with open(f"{BASE_PATH}/Data/supervised/{self.action_type}_0T/sup_y_batch_{j}.pickle", "rb") as data:
                    x_list = pickle.load(data)

            if j == 0:
                x = np.asarray(x_list)
                y = np.asarray(y_list)
            else:
                x = np.concatenate((x, np.asarray(x_list)), axis=0)
                y = np.concatenate((y, np.asarray(y_list)), axis=0)


        if scaling:
            min_max_scaler = preprocessing.MinMaxScaler()
            for j in range(len(x)):
                x[j] = min_max_scaler.fit_transform(x[j])

        
        if self.translation == False:
            y = y[:, 2:]
        #splits data into train, test and validation
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        print(np.shape(self.x_train), np.shape(self.x_test), np.shape(self.x_val))
        print(np.shape(y_train), np.shape(y_test), np.shape(y_val))

       
    def train(self, batch_size):
        """function to train the network

        Args:
            batch_size (int): traiing batch size
        """
        lr_scheduler = StepDecay(
            initial_lr=0.001,
            drop_every=50,
            decay_factor=0.5
        )
        #callback for learnrate sheduling
        lr_callback = keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_scheduler(epoch), verbose=True
        )
        #compiling the model
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.001),
            loss = keras.losses.Huber(),
            metrics=[keras.metrics.MeanSquaredError()],
            run_eagerly=False
        )
        self.model.build(input_shape=(None, 2048, 4))
        print(self.model.summary())
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size = batch_size,
            epochs = self.epochs,
            callbacks=lr_callback,
            validation_data=(self.x_val, self.y_val)
        )
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('Mean Squared Error')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.ylim([0,1])
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim([0,1])
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #plt.close()



    def save_model(self, name):
        """saves network

        Args:
            name (string): name of model
        """
        BASE_PATH = os.path.abspath(os.getcwd())
        self.model.save(f"{BASE_PATH}/Data/supervised/networks/{self.action_type}/{name}")

    def load_model(self, name):
        """loads saved network

        Args:
            name (string): name of model
        """
        BASE_PATH = os.path.abspath(os.getcwd())
        self.model = keras.models.load_model(f"{BASE_PATH}/Data/supervised/networks/{self.action_type}/{name}")
        #self.model = keras.models.load_model(f"{BASE_PATH}/Data/supervised/networks/{self.action_type}/{name}", custom_objects={"OrthogonalRegularizer": OrthogonalRegularizer})

    def evaluate(self, perform_scan):
        """evaluated network based on validation data
        furthermore plots ground truth and predicted cam pose and orientation in the same colors for 10 different configurations
        calculates distance as metric for translation and angle as metric for rotation

        Args:
            perform_scan (bool): if scans should be performed - 10 configurations get plotted

        Returns:
            results: loss values of evaluation data
            metrics: mean value of distance and angle as metric for translation and rotation
        """
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=128)
        
        # if perform_scan = True, only 10 configurations get validated and plotted
        if perform_scan:
            test = 10
        else:
            test = len(self.x_val)
        init = False
        state = self.x_val[:test]
        print(np.shape(state))
        action_true = self.y_val[:test]
        predictions = self.model.predict(np.asarray(state))
        #print(predictions)
        scanned_points = []
        poses = []
        loss_q = []
        dist = []
        for j in range(len(state)):
            if self.translation:
                action_true_con = action_true[j]
            else:
                action_true_con= np.concatenate(([1.42, 0.76], action_true[j]), axis=None)
            action_true_mapped = action_mapping(self.action_type, action_true_con)
            pos, alpha, beta, gamma = get_cam_pose_and_rotation(action_type=self.action_type, actions=action_true_mapped, radius=self.CamRadius, initial=init)
            transform = get_camera_transform_from_euler(pos, alpha, beta, gamma, degrees=False)
            poses.append(transform)
            
            if perform_scan:
                self.scanner.reset()
                self.scanner.setup(self.mesh_path)
                self.scanner.create_encoding_array()
                self.scanner.single_scan(translation=pos, alpha=alpha, beta=beta, gamma=gamma)
                self.scanner.update_binary_state_array()
            
            if self.translation:
                action_predicted = predictions[j]
            else:
                action_predicted = np.concatenate(([1.42, 0.76], predictions[j]), axis=0)
            
            
            
            action_pred = action_mapping(self.action_type, action_predicted)
            pos_2, alpha_2, beta_2, gamma_2 = get_cam_pose_and_rotation(action_type=self.action_type, actions=action_pred, radius=self.CamRadius, initial=init)
            transform_2 = get_camera_transform_from_euler(pos_2, alpha_2, beta_2, gamma_2, degrees=False)
            poses.append(transform_2)
            
            dist.append(np.sqrt((pos_2[0]-pos[0])**2 + (pos_2[1]-pos[1])**2 + (pos_2[2]-pos[2])**2))
            loss_q.append(getAngle(transform[:-1, :-1], transform_2[:-1, :-1]))

            if perform_scan:
                self.scanner.reset()
                self.scanner.setup(self.mesh_path)
                self.scanner.create_encoding_array()
                self.scanner.single_scan(translation=pos, alpha=alpha, beta=beta, gamma=gamma)
                self.scanner.update_binary_state_array()

            
        print("dist: ", np.mean(np.asarray(dist)))

        print("q_loss: ", np.mean(np.asarray(loss_q)))
        
        metrics = [np.mean(np.asarray(dist)), np.mean(np.asarray(loss_q))]
        
        if perform_scan:
            vis_mesh_and_cam_pos(self.mesh_path, poses)
        return results, metrics


def getAngle(P, Q):
    "https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices"
    R = np.dot(P, Q.T)
    theta = (np.trace(R)-1) / 2
    return np.arccos(theta)*(180/np.pi)


if __name__ == "__main__":
    action_type = "2T2R" #action type 
    epochs = 50 #number of epochs to train for
    translation = True #if translation should be used (special case)
    batch_size = 32 #size of batch size for training

    sup = Sup_Framework(action_type=action_type, epochs=epochs, translation=translation)
    #sup.create_dataset()
    sup.load_dataset(scaling=True)
    
    sup.create_network(net_type="Pointnet_NN_Model")

    sup.train(batch_size)
    # sup.save_model("test_2")
    # sup.load_model("test_2")
    results, metrics = sup.evaluate(perform_scan=True)
    print(metrics)


