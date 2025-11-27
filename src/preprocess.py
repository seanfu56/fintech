import numpy as np
from hftbacktest.data.utils import binancefutures
from hftbacktest.data.utils.snapshot import create_last_snapshot

# _ = binancefutures.convert(
#     'data/btcusdt_20251116.gz',
#     output_filename = 'data/btcusdt_20251116.npz',
#     combined_stream = True,
#     buffer_size=500_000_000,
# )

# _ = binancefutures.convert(
#     'data/btcusdt_20251117.gz',
#     output_filename = 'data/btcusdt_20251117.npz',
#     combined_stream = True,
#     buffer_size=500_000_000,
# )

_ = create_last_snapshot(
    ['data/btcusdt_20251116.npz'],
    output_snapshot_filename = 'data/btcusdt_20251116_eod.npz',
    tick_size=0.1,
    lot_size=0.001,
)