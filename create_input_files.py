from utils import create_input_files

if __name__ == "__main__":
    # Create input files (along with word map)
    create_input_files(
        tokenizer_name="monologg/kobigbird-bert-base",
        json_path="../Dataset/MSCOCO_train_val_Korean.json",
        image_folder="../Dataset",
        captions_per_image=5,
        output_folder="../Dataset",
        max_len=50,
    )
