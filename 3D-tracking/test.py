
class ProximityEvent:
    def __init__(self, start_time, person_id):
        self.start_time = start_time
        self.person_id = person_id
        self.status = "active"
        self.hand_images = []
        self.end_time = None

    def get_id(self):
        return str(self.get_person_id()) + '_' + str(self.get_start_time()) + '_' + str(self.get_end_time())

    def get_event_id(self):
        return self.get_id()

    def get_person_id(self):
        return self.person_id

    def get_hand_images(self):
        return self.hand_images

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def set_end_time(self, value):
        self.end_time = value

    def add_hand_images(self, image):
        self.hand_images.append(image)

    def set_status(self, value):
        self.status = value




current_events = {}
tommy = {}
var = ProximityEvent(2, 15)
current_events['1'] = var
tommy['2'] = var
current_events['1'].set_status("completed")
del current_events['1']
print(tommy['2'].status)