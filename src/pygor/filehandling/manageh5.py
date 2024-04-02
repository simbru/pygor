import dacite

def metadata_dict(HDF5_file):
    date, time = get_experiment_datetime(HDF5_file["wParamsStr"])
    metadata_dict = {
    "filename"       : HDF5_file.filename,
    "exp_date"       : date,
    "exp_time"       : time,
    "objectiveXYZ"   : get_rel_objective_XYZ(HDF5_file["wParamsNum"]),
    }
    return metadata_dict


def load_from_hdf5(path):
    """
    Loads an HDF5 file directly and writes it to an object, with keys in HDF5 file 
    becoming attributes of that object. 

    Note that you don't get any of the fancy processing attributes with this, just access to waves,
    to be used only for utility 
    """
    new_dict = {}
    with h5py.File(path) as HDF5_file:
        metadata = metadata_dict(HDF5_file)
        for key in HDF5_file.keys():
            new_dict[key] = np.array(HDF5_file[key]).T ## note rotation
    data_dict = new_dict
    final_dict = (data_dict | metadata)
    @dataclass
    class Data_hdf5:
        # Automatically maps contents of HDF5 file
        __annotations__ = {key: type(data_type) for key, data_type in final_dict.items()}
        def attributes(self):
            return list(self.__annotations__)
    # Dacite is a package that allows you to create DataClass objects from dictionaries
    object = dacite.from_dict(Data_hdf5, final_dict)
    return object