from transformers.cache_utils import DynamicCache
try:
    from transformers.cache_utils import SlidingWindowCache
    print("Found SlidingWindowCache")
except ImportError:
    print("SlidingWindowCache NOT found")
