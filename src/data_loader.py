import os
import h5py
import scipy.io
import numpy as np
import tifffile as tiff


def load_image(full_path):
    ext = os.path.splitext(full_path)[1].lower()

    try:
        if ext in ['.tif', '.tiff']:
            img = tiff.imread(full_path)
            if np.issubdtype(img.dtype, np.unsignedinteger):
                img = np.float64(img)
            return img

        elif ext == '.mat':
            try:
                # Try loading with scipy (works for MATLAB v7.2 and below)
                mat_data = scipy.io.loadmat(full_path, squeeze_me=True, struct_as_record=False)
                if 'result' in mat_data:
                    result = mat_data['result']
                    if hasattr(result, 'corrected_phase'):
                        return np.array(result.corrected_phase, dtype=np.float64)
                    else:
                        print(f"'result' struct found, but 'corrected_phase' field is missing.")
                        return None
                else:
                    print(f"'result' struct not found in the file.")
                    return None

            except NotImplementedError as e:
                # Handle MATLAB v7.3+ files (HDF5 format)
                with h5py.File(full_path, 'r') as f:
                    # Print keys to help debug structure
                    print(f"MAT v7.3 structure keys: {list(f.keys())}")

                    # This assumes structure like /result/corrected_phase
                    if 'result' in f:
                        result_group = f['result']
                        if 'corrected_phase' in result_group:
                            data = result_group['corrected_phase']
                            return np.array(data)
                        else:
                            print(f"'corrected_phase' not found in result group of the file")
                            return None
                    else:
                        print(f"'result' group not found in the file")
                        return None

        else:
            print(f"Unsupported file format: {ext}")
            return None

    except Exception as e:
        print(f"Failed to load file '{full_path}': {e}")
        return None


def load_data(data_paths):
    for index, data_path in enumerate(data_paths):
        phase = load_image(data_path).astype(np.float32)
        phase -= np.median(phase)
        phase = phase[np.newaxis, :, :]  # expand to 3D

        if index == 0:
            data_stack = phase
        else:
            data_stack = np.concatenate((data_stack, phase), axis=0)  # Concatenate along 3rd axis
    print(f"Loaded {data_stack.shape[0]} images of size {data_stack.shape[1:]}")
    return data_stack
