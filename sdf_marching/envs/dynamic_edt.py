from dynamic_edt_mapping import DynamicEDTMapping

class DynamicEDT:
    def __init__(self, filename, unknown_as_occupied=False, max_dist=5.0):
        self.edt_wrapper = DynamicEDTMapping(
            filename, 
            unknown_as_occupied=unknown_as_occupied, 
            max_dist=max_dist
        )

    def __call__(self, test_positions):
        return self.edt_wrapper.query_sdf(test_positions.transpose())

    def point(self):
        pass

    @property
    def mins(self):
        return self.edt_wrapper.mins()
    
    @property
    def maxs(self):
        return self.edt_wrapper.maxs()