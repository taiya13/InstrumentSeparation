"""Backbone registry — lets any model be selected by name from config.

Adding a new model = write a wrapper subclass, decorate with @register("name").
Swapping models is then a one-line config change (`model.name: ...`).
"""
_BACKBONES = {}


def register(name):
    def deco(cls):
        cls.name = name
        _BACKBONES[name] = cls
        return cls
    return deco


def get_backbone(name):
    if name not in _BACKBONES:
        raise KeyError(f"unknown backbone '{name}'. available: {available()}")
    return _BACKBONES[name]


def available():
    return sorted(_BACKBONES)
