"""
@Description: Awesome && Useful python utils collection
@Author: Ken Zh0ng
@date: 2024-06-06
"""
import os
import yaml


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = Dotdict(value)
            self[key] = value


class HyperParam(Dotdict):
    def __init__(self, file):
        super().__init__()
        hp_dict = HyperParam.load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

    @staticmethod
    def load_hparam(file):
        """
        Load hyperparameters from yaml file
        
        Args:
            file: str, path of yaml file
            
        Returns:
            hparam_dict: dict, hyperparameters
        """
        stream = open(file, "r")
        docs = yaml.load_all(stream, Loader=yaml.Loader)
        hparam_dict = dict()
        for doc in docs:
            for k, v in doc.items():
                hparam_dict[k] = v
        return hparam_dict

    @staticmethod
    def load_hparam_str(hp_str):
        """
        Load hyperparameters from yaml string
        
        Args:
            hp_str: str, yaml string
        
        Returns:
            hparam_dict: HyperParam instance, 
        """
        path = "temp-restore.yaml"
        with open(path, "w") as f:
            f.write(hp_str)
        hparam_dict = HyperParam(path)
        os.remove(path)
        return hparam_dict


 



####################################################################
#                      Funcs Only for Test                         #
####################################################################

def test_Dotdict():
    """
    Only for test 'Dotdict' class
    """
    di = {'a':123, 'b':456, 'c':{'d':789}} # nested dict
    dotd = Dotdict(di)
    
    print(dotd.a)
    print(dotd.b)
    print(type(dotd.c), dotd.c.d)
    
    dotd.ff = [123,4,5]
    print(dotd.ff)


if __name__ == "__main__" :
    test_Dotdict()

