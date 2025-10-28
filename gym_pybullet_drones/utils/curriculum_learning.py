import numpy as np


class CurriculumLearning():
    def __init__(self, starting_level=0):
        self.curriculum_level = starting_level
        self.curriculum_success_tally = 0

        
        self.curriculum_levels = {
            #Drone Spacing - Relaxed
            #
            0: {
                #episode_params
                'drone_desired_distance': 0.8,              # meters
                'drone_spacing_tolerance': 0.3,             # %-based
                'drone_spacing_hold_timer': 10,
                'cattle_approach_desired_distance': 0.0,    # meters
                'min_effectiveness': 0.0,                    # out of 100
                'cattle_desired_distance': 0.0,
                'cattle_spacing_tolerance': 0.0, 
                'min_drones': 3,
                'max_drones': 3,
                'episode_length': 40,
                #reward_weightings
                'drone_to_drone_spacing_simple': 1,
                'drone_to_drone_spacing_complex': 0,
                'drone_survival': 0,
                'cattle_approach': 0,
                'effectiveness': 0,
                'cattle_to_drone_spacing': 0.0,
                #success
                'required_tally': 100
            },
            #Drone Spacing - Strict
            1: {
                #episode_params
                'drone_desired_distance': 0.8,
                'drone_spacing_tolerance': 0.2,             # %-based
                'drone_spacing_hold_timer': 25,
                'cattle_approach_min_distance': 0.0,        # meters
                'min_effectiveness': 0.0,                    # out of 100
                'cattle_desired_distance': 0.0,
                'cattle_spacing_tolerance': 0.0,   
                'min_drones': 4,
                'max_drones': 4,
                'episode_length': 40,
                #reward_weightings
                'drone_to_drone_spacing_simple': 0,
                'drone_to_drone_spacing_complex': 1,
                'drone_survival': -0.5,
                'cattle_approach': 0,
                'effectiveness': 0,
                'cattle_to_drone_spacing': 0.0,
                #success
                'required_tally': 300
            },
            #Cattle Approach - Relaxed
            2: {
                #episode_params
                'drone_desired_distance': 0.8,
                'drone_spacing_tolerance': 0.2,             # %-based
                'drone_spacing_hold_timer': 15,
                'cattle_approach_min_distance': 0.6,        # meters
                'min_effectiveness': 0.0,                    # out of 100
                'cattle_desired_distance': 0.0,
                'cattle_spacing_tolerance': 0.0,  
                'min_drones': 4,
                'max_drones': 4,
                'episode_length': 40, 
                #reward_weightings
                'drone_to_drone_spacing_simple': 0,
                'drone_to_drone_spacing_complex': 0.8,
                'drone_survival': 0,
                'cattle_approach': 1,
                'effectiveness': 0,
                'cattle_to_drone_spacing': 0.0,
                #success
                'required_tally': 100
            },
            #Cattle Approach - Strict
            3: {
                #episode_params
                'drone_desired_distance': 0.8,
                'drone_spacing_tolerance': 0.2,             # %-based
                'drone_spacing_hold_timer': 15,
                'cattle_approach_min_distance': 0.3,        # meters
                'min_effectiveness': 0.0,                    # out of 100
                'cattle_desired_distance': 0.0,
                'cattle_spacing_tolerance': 0.0,     
                'min_drones': 4,
                'max_drones': 4,
                'episode_length': 40,
                #reward_weightings
                'drone_to_drone_spacing_simple': 0,
                'drone_to_drone_spacing_complex': 0.8,
                'drone_survival': -0.5,
                'cattle_approach': 1,
                'effectiveness': 0,
                'cattle_to_drone_spacing': 0.0,
                #success
                'required_tally': 400
            },
            #Surrounding Cattle
            4: {
                #episode_params
                'drone_desired_distance': 0.8,
                'drone_spacing_tolerance': 0.2,             # %-based
                'drone_spacing_hold_timer': 15,
                'cattle_approach_min_distance': 0.3,        # meters
                'min_effectiveness': 20,                    # out of 100  
                'cattle_desired_distance': 0.0,
                'cattle_spacing_tolerance': 0.0, 
                'min_drones': 4,
                'max_drones': 4,
                'episode_length': 80,      
                #reward_weightings
                'drone_to_drone_spacing_simple': 0,
                'drone_to_drone_spacing_complex': 0.7,
                'drone_survival': -0.0,
                'cattle_approach': 0.8,
                'effectiveness': 1,
                'cattle_to_drone_spacing': 0.0,
                #success
                'required_tally': 600
            },
            #Cattle Spacing
            5: {
                #episode_params
                'drone_desired_distance': 0.8,
                'drone_spacing_tolerance': 0.2,             # %-based
                'drone_spacing_hold_timer': 15,
                'cattle_approach_min_distance': 0.3,        # meters
                'min_effectiveness': 50,                    # out of 100    
                'cattle_desired_distance': 0.8,
                'cattle_spacing_tolerance': 0.1,  
                'min_drones': 4,
                'max_drones': 4,
                'episode_length': 40,      
                #reward_weightings
                'drone_to_drone_spacing_simple': 0,
                'drone_to_drone_spacing_complex': 0.7,
                'drone_survival': -0.5,
                'cattle_approach': 0.6,
                'effectiveness': 1,
                'cattle_to_drone_spacing': 0.8,
                #success
                'required_tally': 600
            },
            6: {#Test curroiulum
                #episode_params
                'drone_desired_distance': 0.8,
                'drone_spacing_tolerance': 0.3,             # %-based
                'drone_spacing_hold_timer': 15,
                'cattle_approach_min_distance': 0.2,        # meters
                'min_effectiveness': 50,                    # out of 100  
                'cattle_desired_distance': 0.0,
                'cattle_spacing_tolerance': 0.0, 
                'min_drones': 4,
                'max_drones': 12,
                'episode_length': 80,      
                #reward_weightings
                'drone_to_drone_spacing_simple': 0.7,
                'drone_to_drone_spacing_complex': 0.0,
                'drone_survival': -0.0,
                'cattle_approach': 0.8,
                'effectiveness': 1,
                'cattle_to_drone_spacing': 0.0,
                #success
                'required_tally': 600
            },
            7: {#Test curroiulum
                #episode_params
                'drone_desired_distance': 0.8,
                'drone_spacing_tolerance': 0.3,             # %-based
                'drone_spacing_hold_timer': 15,
                'cattle_approach_min_distance': 0.2,        # meters
                'min_effectiveness': 50,                    # out of 100  
                'cattle_desired_distance': 0.0,
                'cattle_spacing_tolerance': 0.0, 
                'min_drones': 4,
                'max_drones': 12,
                'episode_length': 80,      
                #reward_weightings
                'drone_to_drone_spacing_simple': 0.0,
                'drone_to_drone_spacing_complex': 0.0,
                'drone_survival': -0.0,
                'cattle_approach': 1,
                'effectiveness': 1,
                'cattle_to_drone_spacing': 0.0,
                #success
                'required_tally': 600
            },
        }

        self.current_curriculum = self.curriculum_levels[self.curriculum_level]
    


    def evaluate_curriculum_results(self, episode_result: bool):
        '''
        See current results (true if epiosde temrinated) level up learning if results are achieved
        '''
        if episode_result:
            self.curriculum_success_tally += 1
            curriculum_data = self.curriculum_levels[self.curriculum_level]
            required_success_tally = curriculum_data['required_tally']
            print(f"Level Passed! Curriculum Score: {self.curriculum_success_tally} / {curriculum_data['required_tally']}")

            if self.curriculum_success_tally >= required_success_tally:
                
                self.curriculum_success_tally = 0
                self.curriculum_level += 1
                if self.curriculum_level >= len(self.curriculum_levels):
                    print("LEARNING IS DONE!!!!!")
                    self.curriculum_level = len(self.curriculum_levels) - 1
                    return

                self.current_curriculum = self.curriculum_levels[self.curriculum_level]
    



