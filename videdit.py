import argparse
import datetime
import json
import random
import os
import re
import numpy as np
import cv2

from os.path import isfile, join
from subprocess import run, PIPE
from keras.models import model_from_json
from matplotlib import pyplot as plot


def _smooth(x, window_len=4, window='flat'):
    if len(x) < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # y = y[-len(x):]

    d = len(y) - len(x)
    s = int(len(y) / d)
    i = 0
    l = []
    for _ in range(d):
        l.append(i)
        i += s

    y = np.delete(y, l)

    return y


def get_json_dict(filename):
    f = open(filename)
    data = json.load(f)
    f.close()
    return data


def get_file_content(filename):
    f = open(filename)
    data = f.read()
    f.close()
    return data


def get_duration(media_filename):
    c = run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', media_filename],
            stderr=PIPE, stdout=PIPE)
    if c.returncode != 0:
        stdout = c.stdout.decode('utf-8')
        stderr = c.stderr.decode('utf-8')
        raise ChildProcessError(
            'ffprobe exited with error code {}\nstdout:\n{}\n\nstderr:\n{}'.format(c.returncode, stdout, stderr))
    s = c.stdout.decode("utf-8")
    j = json.loads(s)
    return float(j['format']['duration'])


def extract_and_get_frames(media_filename, interval, tmpdir, randid):
    randid = '{:05d}'.format(randid)
    basename = os.path.basename(media_filename)
    basename, ext = os.path.splitext(basename)
    fullbasename = '{}-id{}-seq%05d.jpg'.format(basename, randid)
    fullbasename = join(tmpdir, fullbasename)
    c = run(['ffmpeg', '-i', media_filename, '-vf', 'fps=1/{}'.format(interval), fullbasename])
    if c.returncode != 0:
        stdout = c.stdout.decode('utf-8')
        stderr = c.stderr.decode('utf-8')
        raise ChildProcessError(
            'ffmpeg exited with error code {}\nstdout:\n{}\n\nstderr:\n{}'.format(c.returncode, stdout, stderr))
    regex = '({})(-id{})(-seq)([0-9])+(.jpg)'.format(basename, randid)
    return [join(tmpdir, f) for f in os.listdir(tmpdir) if re.match(regex, f)]


def score_frames(frames, model, mean, std, size):
    result = []
    for frame in frames:
        img = cv2.imread(frame)
        img = cv2.resize(img, (size, size))
        img = np.array(img, np.float32) / 255.
        img -= mean
        img /= std
        img = img.reshape((1,) + img.shape)
        pred = model.predict(img)
        result.append(pred[0][0])
        del img
    return result


def delete_files(files):
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


def get_scores_smooth(y):
    return _smooth(y)


def plot_boundaries(original, smoothed, thres):
    timeserie = []
    threshold = []
    for i, score in enumerate(original):
        timeserie.append(i)
        threshold.append(thres)
    plot.plot(timeserie, original, '-r')
    plot.plot(timeserie, smoothed, '-g')
    plot.plot(timeserie, threshold, '-k')
    plot.show()


def get_cuts(x, threshold):
    cuts = []
    ini = -1
    for i in range(len(x)):
        if ini == -1:
            if x[i] > threshold:
                ini = i
        else:
            if x[i] < threshold:
                cuts.append((ini, i))
                ini = -1
    return cuts


def cut_and_join(media_filename, cuts, interval, tmpdir):
    cut_names = []
    basefilename, ext = os.path.splitext(media_filename)
    for i, (begin, end) in enumerate(cuts):
        begin_sec = begin * interval
        end_sec = end * interval
        begin_time = str(datetime.timedelta(seconds=begin_sec))
        run_time = str(datetime.timedelta(seconds=end_sec - begin_sec))
        fullname = basefilename + '-{:04d}{}'.format(i, ext)
        run(['ffmpeg', '-i', media_filename, '-acodec', 'copy', '-vcodec', 'copy', '-ss', begin_time, '-t', run_time,
             fullname])
        cut_names.append(fullname)

    if len(cut_names) > 0:
        listfilename = os.path.basename(media_filename)
        listfilename = join(tmpdir, listfilename)
        listfilename = listfilename + '-joinlist.txt'
        try:
            f = open(listfilename, 'w')
            for cutname in cut_names:
                f.write('file ' + '\'{}\''.format(cutname) + os.linesep)
            f.close()
            editedfilename = basefilename + '-edited' + ext
            run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', '{}'.format(listfilename), '-c', 'copy', editedfilename])
            return editedfilename
        finally:
            delete_files(cut_names)
            if os.path.isfile(listfilename):
                os.remove(listfilename)


def process_media(model, media_filename, config, plot, smooth):
    std = config['std']
    mean = config['mean']
    threshold = config['threshold']
    interval = config['interval']
    image_size = config['image_size']
    tmpdir = config['tmp_dir']

    # duration = get_duration(media_filename)

    randid = random.randint(0, 99999)
    frames = extract_and_get_frames(media_filename, interval, tmpdir, randid)
    try:
        scores_original = score_frames(frames, model, mean, std, image_size)
        scores_smoothed = get_scores_smooth(scores_original)
        if smooth:
            cuts = get_cuts(scores_smoothed, threshold)
        else:
            cuts = get_cuts(scores_original, threshold)

        print(cuts)

        if plot:
            plot_boundaries(scores_original, scores_smoothed, threshold)

        print(scores_original)
        print(scores_smoothed)

        return cut_and_join(media_filename, cuts, interval, tmpdir)
    finally:
        delete_files(frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--media', type=str, default='', metavar='N', help='The Media filename for editing')
    parser.add_argument('--model', type=str, default='', metavar='N',
                        help='The Keras model used for frame based classification')
    parser.add_argument('--weights', type=str, default='', metavar='N',
                        help='The Keras model\'s weights used for classification')
    parser.add_argument('--config', type=str, default='', metavar='N', help='The config for this session')
    parser.add_argument('--histo', action='store_true', default=False,
                        help='Plot time series of frame scores, useful to predict how the media will be edited')
    parser.add_argument('--no-smooth', action='store_true', default=False,
                        help='Whether to smooth the predictions or not')
    args = parser.parse_args()

    media_filename = args.media
    model_filename = args.model
    weights_filename = args.weights
    config_filename = args.config
    plot_histogram = args.histo
    smooth = not args.no_smooth

    if media_filename == '':
        raise ValueError('Use --media <filename> to provide media filename')

    if not isfile(media_filename):
        raise ValueError('Media file "{}" not found'.format(media_filename))

    if model_filename == '':
        raise ValueError('Use --model <filename> to provide model used for classification')

    if not isfile(model_filename):
        raise ValueError('Model file "{}" not found'.format(model_filename))

    if weights_filename == '':
        raise ValueError('Use --weights <filename> to provide model\'s weights')

    if not isfile(weights_filename):
        raise ValueError('Weights file "{}" not found'.format(weights_filename))

    if config_filename == '':
        raise ValueError('Use --config <filename> to provide configuration filename')

    if not isfile(config_filename):
        raise ValueError('Config file "{}" not found'.format(config_filename))

    config = get_json_dict(config_filename)

    model_json = get_file_content(model_filename)
    model = model_from_json(model_json)
    model.load_weights(weights_filename)

    outfile = process_media(model, media_filename, config, plot_histogram, smooth)

    print('file save: {}'.format(outfile))
