import gzip
import tarfile
import json
import numpy as np
from hftbacktest import DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, EXCH_EVENT, LOCAL_EVENT, BUY_EVENT, SELL_EVENT

def convert_okx(input_file, output_filename):
    print(f"Converting {input_file} to {output_filename}")
    
    # Pre-allocate a large array, we'll trim it later
    # Estimate size: 10 million rows should be enough for a day of high frequency data, 
    # but let's use a dynamic list and convert to array at the end to be safe, 
    # or better, use a large pre-allocated array and resize.
    # Given the file size (~400MB compressed), 10M events is a reasonable upper bound estimate.
    # Each event has 8 fields (ev, exch_ts, local_ts, px, qty, order_id, ival, fval)
    # We'll use a list of lists for simplicity and convert to structured array at the end,
    # or write chunks. For simplicity and performance with numba/numpy later, let's build a structured array.
    
    # Define the dtype for hftbacktest
    dtype = [
        ('ev', '<u8'),
        ('exch_ts', '<i8'),
        ('local_ts', '<i8'),
        ('px', '<f8'),
        ('qty', '<f8'),
        ('order_id', '<u8'),
        ('ival', '<i8'),
        ('fval', '<f8')
    ]
    
    data_list = []
    
    with tarfile.open(input_file, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f:
                    # OKX data is often in a file inside the tar
                    # The file inside might be text or another archive? 
                    # Based on previous `tar -tzf`, it's `BTC-USDT-SWAP-L2orderbook-400lv-2025-02-01.data`
                    # And `head` showed it's JSON lines.
                    
                    content = f.read()
                    # It might be large, so reading line by line is better if possible, 
                    # but extractfile returns a file-like object.
                    
                    # We need to handle the fact that we read the whole file into memory if we do f.read()
                    # Let's iterate over lines
                    f.seek(0)
                    
    data_list = []
    chunk_size = 100000
    all_chunks = []
    
    with tarfile.open(input_file, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                if f:
                    # Iterate over lines
                    f.seek(0)
                    
                    line_count = 0
                    for line in f:
                        line_count += 1
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                            
                        # Extract fields
                        try:
                            ts = int(obj['ts'])
                            ts_ns = ts * 1000 * 1000 
                            
                            action = obj.get('action', '')
                            
                            event_type = DEPTH_EVENT
                            if action == 'snapshot':
                                event_type = DEPTH_SNAPSHOT_EVENT
                            
                            # Process Asks
                            for ask in obj.get('asks', []):
                                px = float(ask[0])
                                qty = float(ask[1])
                                
                                # Ask = SELL side
                                ev = event_type | EXCH_EVENT | LOCAL_EVENT | SELL_EVENT
                                
                                data_list.append((
                                    ev,
                                    ts_ns,
                                    ts_ns, # local_ts same as exch_ts for now
                                    px,
                                    -qty, # Ask is negative
                                    0, # order_id
                                    0, # ival
                                    0.0 # fval
                                ))
                                
                            # Process Bids
                            for bid in obj.get('bids', []):
                                px = float(bid[0])
                                qty = float(bid[1])
                                
                                # Bid = BUY side
                                ev = event_type | EXCH_EVENT | LOCAL_EVENT | BUY_EVENT
                                
                                data_list.append((
                                    ev,
                                    ts_ns,
                                    ts_ns,
                                    px,
                                    qty, # Bid is positive
                                    0,
                                    0,
                                    0.0
                                ))
                        except Exception as e:
                            print(f"Error parsing line {line_count}: {e}")
                            print(f"Line content: {line}")
                            continue

                        if len(data_list) >= chunk_size:
                            try:
                                chunk_arr = np.array(data_list, dtype=dtype)
                                all_chunks.append(chunk_arr)
                                data_list = []
                                print(f"Processed {line_count} lines...")
                            except Exception as e:
                                print(f"Error converting chunk at line {line_count}: {e}")
                                # Debug the chunk
                                for i, item in enumerate(data_list):
                                    try:
                                        np.array([item], dtype=dtype)
                                    except:
                                        print(f"Bad item at index {i}: {item}")
                                        raise e

    # Process remaining
    if data_list:
        try:
            chunk_arr = np.array(data_list, dtype=dtype)
            all_chunks.append(chunk_arr)
        except Exception as e:
            print(f"Error converting last chunk: {e}")
            for i, item in enumerate(data_list):
                try:
                    np.array([item], dtype=dtype)
                except:
                    print(f"Bad item at index {i}: {item}")
                    raise e

    # Concatenate all chunks
    if all_chunks:
        data = np.concatenate(all_chunks)
    else:
        data = np.array([], dtype=dtype)
    
    # Sort
    data = np.sort(data, order='exch_ts')
    
    # Save
    np.savez_compressed(output_filename, data=data)
    print(f"Saved {len(data)} events to {output_filename}")

if __name__ == '__main__':
    # Convert the files we found
    # data/okx/BTC-USDT-SWAP-L2orderbook-400lv-2025-02-01.tar.gz
    
    convert_okx(
        'data/okx/BTC-USDT-SWAP-L2orderbook-400lv-2025-02-01.tar.gz',
        'data/convert/btcusdt_20250201.npz'
    )
    
    # We can also convert the second file if needed, but let's start with one.
