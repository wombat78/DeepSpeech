import fnmatch
import os
import random
import subprocess
import tarfile
import tensorflow as tf
import unicodedata
import codecs
import pandas

from glob import glob
from itertools import cycle
from math import ceil
from sox import Transformer
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from threading import Thread
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import text_to_char_array, ctc_label_dense_to_sparse

class DataSets(object):
    def __init__(self, train, dev, test):
        self._dev = dev
        self._test = test
        self._train = train

    def start_queue_threads(self, session):
        self._dev.start_queue_threads(session)
        self._test.start_queue_threads(session)
        self._train.start_queue_threads(session)

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test

class DataSet(object):
    def __init__(self, filelist, thread_count, batch_size, numcep, numcontext):
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None,])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None,], []],
                                                  dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity=2 * self._get_device_count() * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._filelist = filelist
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_circular_list = self._create_files_circular_list()

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return max(len(available_gpus), 1)

    def start_queue_threads(self, session):
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
        return batch_threads

    def _create_files_circular_list(self):
        # 1. Sort by wav filesize
        # 2. Select just wav filename and transcript columns
        # 3. Create a cycle
        return cycle(self._filelist.sort_values(by="wav_filesize")
                                   .ix[:, ["wav_filename", "transcript"]]
                                   .itertuples(index=False))

    def _populate_batch_queue(self, session):
        for wav_file, transcript in self._files_circular_list:
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            target = text_to_char_array(transcript)
            target_len = len(target)
            try:
                session.run(self._enqueue_op, feed_dict={
                    self._x: source,
                    self._x_length: source_len,
                    self._y: target,
                    self._y_length: target_len})
            except (RuntimeError, tf.errors.CancelledError):
                return

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._example_queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_filelist) % _batch_size != 0, this re-uses initial files
        return int(ceil(float(len(self._filelist)) /float(self._batch_size)))


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=8, limit_dev=0, limit_test=0, limit_train=0):
    # Read the processed set files from disk if they exist, otherwise create
    # them.
    train_files = None
    train_csv = os.path.join(data_dir, "librivox-train.csv")
    if gfile.Exists(train_csv):
        train_files = pandas.read_csv(train_csv)

    dev_files = None
    dev_csv = os.path.join(data_dir, "librivox-dev.csv")
    if gfile.Exists(dev_csv):
        dev_files = pandas.read_csv(dev_csv)

    test_files = None
    test_csv = os.path.join(data_dir, "librivox-test.csv")
    if gfile.Exists(test_csv):
        test_files = pandas.read_csv(test_csv)

    if train_files is None or dev_files is None or test_files is None:
        train_files, dev_files, test_files = _download_and_process_corpus()

        # Write sets to disk as CSV files
        train_files.to_csv(train_csv)
        dev_files.to_csv(dev_csv)
        test_files.to_csv(test_csv)

    # Create train DataSet from all the train archives
    train = _create_data_set(train_filelist, thread_count, train_batch_size, numcep, numcontext, limit=limit_train)

    # Create dev DataSet from all the dev archives
    dev = _create_data_set(dev_filelist, thread_count, dev_batch_size, numcep, numcontext, limit=limit_dev)

    # Create test DataSet from all the test archives
    test = _create_data_set(test_filelist, thread_count, test_batch_size, numcep, numcontext, limit=limit_test)

    # Return DataSets
    return DataSets(train, dev, test)

def _download_and_process_corpus():
    # Check if we can convert FLAC with SoX before we start
    sox_help_out = subprocess.check_output(["sox", "-h"])
    if sox_help_out.find("flac") == -1:
        print("Error: SoX doesn't support FLAC. Please install SoX with FLAC support and try again.")
        exit(1)

    # Conditionally download data to data_dir
    train_clean_100 = base.maybe_download("train-clean-100.tar.gz", data_dir, "http://www.openslr.org/resources/12/train-clean-100.tar.gz")
    train_clean_360 = base.maybe_download("train-clean-360.tar.gz", data_dir, "http://www.openslr.org/resources/12/train-clean-360.tar.gz")
    train_other_500 = base.maybe_download("train-other-500.tar.gz", data_dir, "http://www.openslr.org/resources/12/train-other-500.tar.gz")

    dev_clean = base.maybe_download("dev-clean.tar.gz", data_dir, "http://www.openslr.org/resources/12/dev-clean.tar.gz")
    dev_other = base.maybe_download("dev-other.tar.gz", data_dir, "http://www.openslr.org/resources/12/dev-other.tar.gz")

    test_clean = base.maybe_download("test-clean.tar.gz", data_dir, "http://www.openslr.org/resources/12/test-clean.tar.gz")
    test_other = base.maybe_download("test-other.tar.gz", data_dir, "http://www.openslr.org/resources/12/test-other.tar.gz")

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    LIBRIVOX_DIR = "LibriSpeech"
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-360"), train_clean_360)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)

    # Conditionally convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    _maybe_convert_wav(work_dir, "train-clean-100", "train-clean-100-wav")
    _maybe_convert_wav(work_dir, "train-clean-360", "train-clean-360-wav")
    _maybe_convert_wav(work_dir, "train-other-500", "train-other-500-wav")

    _maybe_convert_wav(work_dir, "dev-clean", "dev-clean-wav")
    _maybe_convert_wav(work_dir, "dev-other", "dev-other-wav")

    _maybe_convert_wav(work_dir, "test-clean", "test-clean-wav")
    _maybe_convert_wav(work_dir, "test-other", "test-other-wav")

    # Conditionally split LibriSpeech transcriptions
    train_filelist = _maybe_split_transcriptions(work_dir, "train-clean-100", "train-clean-100-wav")
    train_filelist = train_filelist.append(_maybe_split_transcriptions(work_dir, "train-clean-360", "train-clean-360-wav"))
    train_filelist = train_filelist.append(_maybe_split_transcriptions(work_dir, "train-clean-500", "train-clean-500-wav"))

    dev_filelist = _maybe_split_transcriptions(work_dir, "dev-clean", "dev-clean-wav")
    dev_filelist = dev_filelist.append(_maybe_split_transcriptions(work_dir, "dev-other", "dev-other-wav"))

    test_filelist = _maybe_split_transcriptions(work_dir, "test-clean", "test-clean-wav")
    test_filelist = test_filelist.append(_maybe_split_transcriptions(work_dir, "test-other", "test-other-wav"))

    return train_filelist, dev_filelist, test_filelist

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()
        # os.remove(archive)

def _maybe_convert_wav(data_dir, extracted_data, converted_data):
    source_dir = os.path.join(data_dir, extracted_data)
    target_dir = os.path.join(data_dir, converted_data)

    # Conditionally convert FLAC files to wav files
    if not gfile.Exists(target_dir):
        # Create target_dir
        os.makedirs(target_dir)

        # Loop over FLAC files in source_dir and convert each to wav
        for root, dirnames, filenames in os.walk(source_dir):
            for filename in fnmatch.filter(filenames, "*.flac"):
                flac_file = os.path.join(root, filename)
                wav_filename = os.path.splitext(os.path.basename(flac_file))[0] + ".wav"
                wav_file = os.path.join(target_dir, wav_filename)
                transformer = Transformer()
                transformer.build(flac_file, wav_file)
                os.remove(flac_file)

def _maybe_split_transcriptions(extracted_dir, data_set, wav_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    wav_dir = os.path.join(extracted_dir, wav_dir)

    # Loop over transcription files and add entries to the filelist for this split
    #
    # The format for each file 1-2.trans.txt is:
    #  1-2-0 transcription of 1-2-0.flac
    #  1-2-1 transcription of 1-2-1.flac
    #  ...
    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.trans.txt"):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", encoding="utf-8") as fin:
                for line in fin:
                    first_space = line.find(" ")
                    wav_file = line[:first_space] + ".wav"
                    wav_file = os.path.join(wav_dir, wav_file)
                    wav_filesize = os.path.getsize(wav_file)
                    transcript = line[first_space+1:].lower().strip()
                    transcript = unicodedata.normalize("NFKD", transcript).encode("ascii", "ignore")
                    files.append((wav_file, wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

def _create_data_set(filelist, thread_count, batch_size, numcep, numcontext, limit=0):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    # Return DataSet
    return DataSet(filelist, thread_count, batch_size, numcep, numcontext)
