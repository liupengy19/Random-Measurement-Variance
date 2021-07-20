import pstats
from pstats import SortKey
p = pstats.Stats('stats')
p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats(SortKey.TIME)
p.print_stats()