from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px


def main():
    coords_file = './data_in/1111_l123r123_q1_md20_ws15_fixed.csv'
    coords_data = np.loadtxt(coords_file, delimiter=',')

    T = 8000
    class_size = 6
    time_size, reservoir_size = coords_data.shape

    # l123r123 to lr1lr2lr3
    coords_data = np.vstack((coords_data[:9315], coords_data[27585:36900], coords_data[9315:18360],
                             coords_data[36900:46245], coords_data[18360:27585], coords_data[46245:]))
    end_frames = [9315, 18630, 27675, 37020, 46245, 55830]

    # data preprocessing
    X = np.ones((1, reservoir_size))
    start = 0
    for end in end_frames:
        X = np.vstack((X, coords_data[start:start + T]))
        start = end
    X = X[1:]

    class_label = np.array([f'class {i}' for i in range(class_size) for j in range(T)])

    dt = datetime.now()
    prev_title = f'{"{:02d}{:02d}-{:02d}{:02d}".format(dt.month, dt.day, dt.hour, dt.minute)}_{coords_file.split("/")[-1][:-4]}'

    units = [97,156,183]
    attractor = np.hstack([X[:, i-1][:, np.newaxis] for i in units])
    df = pd.DataFrame(attractor, columns=[f'unit {u}' for u in units])
    df['class'] = class_label

    fig = px.scatter_3d(df, x=f'unit {units[0]}', y=f'unit {units[1]}', z=f'unit {units[2]}', color='class')
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        height=400,
        width=600,
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0.5, y=0, z=-0.3),
            eye=dict(x=2, y=2, z=1)
        )
    )
    fig.write_image(f'./image_out/{prev_title}_attractor.eps')
    fig.show()


if __name__ == '__main__':
    main()
