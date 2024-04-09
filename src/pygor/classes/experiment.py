from dataclasses import dataclass
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
# Local imports

# Dependencies


@dataclass
class Experiment:
    data_types = []
    def __str__(self):
        return "MyClass([])"
    def __repr__(self, string = data_types):
        return f"Experiment({string})"
    def __post_init__(self):
        self.data_types : []    
    @classmethod
    def attach_data(self, obj, assumptions = True):        
        """
        TODO Here there should be several tests that check 
        the plane number, rec number, date, and ROI number (more?) 
        with a reference set 
        """       
        def _insert(obj_instance):
            setattr(self, obj_instance.type, obj)
            if obj_instance.type not in self.data_types:
                self.data_types.append(obj_instance.type)
        # self.__dict__[obj.type] = obj
        if isinstance(obj, Iterable):
            raise AttributeError("'obj' must be single ")
            # for i in obj:
                # _insert(i)
        else:
            _insert(obj)
            # self.__repr__ = "ooglie"
        return None

    def detach_data(self, obj):
        del(self.__dict__[obj.type])
        self.data_types.remove(obj.type)
        # return None

    def set_ref(self, obj):
        return None
    def change_ref(self, obj):
        return None 
    def clear_ref(self, obj):
        return None
