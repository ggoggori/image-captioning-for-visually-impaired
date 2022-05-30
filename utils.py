import os
from re import I
import numpy as np
import h5py
import json
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from PIL import Image
from random import seed, choice, sample


def create_input_files(
    tokenizer_name,
    json_path,
    image_folder,
    captions_per_image,
    output_folder,
    max_len=100,
):
    """
    Creates input files for training, validation, and test data.

    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read Karpathy JSON
    with open(json_path, "r") as j:
        data = json.load(j)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []

    for idx, row in enumerate(data):
        if "val" in row["file_path"]:  # ko_coco에는 split key가 없기 때문에 만들어줌.
            if idx >= 20252:
                row["split"] = "test"
            else:
                row["split"] = "valid"
        else:
            row["split"] = "train"

        captions = []

        for cap in row["caption_ko"]:
            if len(tokenizer.tokenize(cap)) <= max_len:
                captions.append(cap)

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, row["file_path"])

        if row["split"] in {"train"}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif row["split"] in {"valid"}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif row["split"] in {"test"}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create a base/root name for all output files
    base_filename = str(captions_per_image) + "_cap_per_img_"

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [
        (train_image_paths, train_image_captions, "TRAIN"),
        (val_image_paths, val_image_captions, "VAL"),
        (test_image_paths, test_image_captions, "TEST"),
    ]:

        with h5py.File(
            os.path.join(
                output_folder, split + "_IMAGES_" + base_filename + ".hdf5"
            ),
            "a",
        ) as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs["captions_per_image"] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset(
                "images", (len(impaths), 3, 256, 256), dtype="uint8"
            )

            print(
                "\nReading %s images and captions, storing to file...\n" % split
            )

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [
                        choice(imcaps[i])
                        for _ in range(captions_per_image - len(imcaps[i]))
                    ]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = read_image(impaths[i])

                # Save image to HDF5 file
                images[i] = img
                # TODO 지금은 MaxLength기준으로 Padding을 수행하는데, 이러면 Padding이 넘 많아짐!! batch별로 padding 하도록 수정하기
                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = tokenizer.encode(
                        c,
                        max_length=max_len,
                        padding="max_length",
                        truncation=True,
                    )
                    # Find caption lengths

                    enc_captions.append(enc_c)
                    caplens.append(len(enc_c) - enc_c.count(0))

            # Sanity check
            assert (
                images.shape[0] * captions_per_image
                == len(enc_captions)
                == len(caplens)
            )

            # Save encoded captions and their lengths to JSON files
            with open(
                os.path.join(
                    output_folder,
                    split + "_CAPTIONS_" + base_filename + ".json",
                ),
                "w",
            ) as j:
                json.dump(enc_captions, j)

            with open(
                os.path.join(
                    output_folder, split + "_CAPLENS_" + base_filename + ".json"
                ),
                "w",
            ) as j:
                json.dump(caplens, j)


def read_image(image_path):
    img = Image.open(image_path)
    img = np.array(img.resize((256, 256)))
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255

    return img


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, "r") as f:
        emb_dim = len(f.readline().split(" ")) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, "r"):
        line = line.split(" ")

        emb_word = line[0]
        embedding = list(
            map(
                lambda t: float(t),
                filter(lambda n: n and not n.isspace(), line[1:]),
            )
        )

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(
    data_name,
    epoch,
    epochs_since_improvement,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    bleu4,
    is_best,
):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "bleu-4": bleu4,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    filename = "checkpoint_" + data_name + ".pth.tar"
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, "BEST_" + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
