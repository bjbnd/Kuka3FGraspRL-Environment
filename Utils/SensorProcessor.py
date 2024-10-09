class SensorProcessor():

    def __init__(self, previous_sensor_data):
    
        self.happened_events = set()
        self.previous_sensor_data = previous_sensor_data
        self.result = False
        self.Event = 0

    def process_sensor_data(self, new_sensor_data):
        # Finding the subtraction between new and previous sensor data
        self.result = False
        self.Event = []
        new_sensor_data = [float(value) for value in new_sensor_data]
        self.previous_sensor_data = [float(value) for value in self.previous_sensor_data]
        subtracted_sensor_data = [cur - prev for cur, prev in zip(new_sensor_data, self.previous_sensor_data)]
        
        for index in range(len(subtracted_sensor_data)):
            if (subtracted_sensor_data[index] == 1) and (index not in self.happened_events):
                ## when the subtraction between the cur and prev is one, means sth has happened,
                ## if the event has not happened before, then the action must be excuted
                self.happened_events.add(index)
                self.previous_sensor_data = new_sensor_data
                self.result = True
                self.Event.append(index)
                
        return [self.result, self.Event]