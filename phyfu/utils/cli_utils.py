import argparse


MODULE_CHOICES = ["brax", "nimble", "taichi", "warp"]
MODEL_NAME_CHOICES = ["two_balls", "ur5e", "catapult", "snake", "mpm", "atlas"]


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", choices=MODULE_CHOICES)
    parser.add_argument("model_name", choices=MODEL_NAME_CHOICES)
    return parser
