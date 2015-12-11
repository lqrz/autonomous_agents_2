__author__ = 'Fritjof'
from collections import defaultdict

class Value_dict:
    def __init__(self, default_value=0.0):
        self.default_value = default_value
        self._values = defaultdict(lambda: self.default_value)

    def get_value(self, state, action):
        return self._values[state.rep(), action]

    def __getitem__(self, item):
        """
        call like this: value = vd[s,a]
        :param item: (state, action) tuple
        :return: value
        """
        return self.get_value(state=item[0], action=item[1])

    def set_value(self, state, action, value):
        self._values[state.rep(), action] = value

    def __setitem__(self, key, value):
        """
        call like this: vd[s,a] = value
        :param key: (state, action) tuple
        :param value: value
        """
        self.set_value(state=key[0], action=key[1], value=value)