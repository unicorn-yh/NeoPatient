import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3 # Used by InceptionScore
from torchvision import transforms
from PIL import Image
# from scipy.stats import entropy # We will implement KL divergence manually with torch for consistency in InceptionScore

# CLIP-specific import
from clip import load, tokenize # Assuming 'clip' library is installed

# FID-specific import
from pytorch_fid import fid_score # Assuming 'pytorch-fid' library is installed

# LPIPS-specific import
import lpips # Assuming 'lpips' library is installed (pip install lpips)

import argparse
# from torch.utils.data import DataLoader, Dataset # Not used in the final combined script
# from tqdm import tqdm # Not used in the final combined script

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # set your own available cuda devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")) # Use .get for safety
print("Torch version:",torch.__version__)
if torch.cuda.is_available():
    print("Cuda version:",torch.version.cuda)
    try:
        print("Cuda name:",torch.cuda.get_device_name(0))
    except Exception as e:
        print(f"Could not get CUDA device name: {e}")
else:
    print("CUDA is not available.")


def get_image_caption(test_metadata_json_path):
    """
    Reads a JSONL metadata file and extracts image filenames and their corresponding captions.

    Args:
        test_metadata_json_path (str): Path to the metadata.jsonl file.

    Returns:
        dict: A dictionary mapping image filenames (e.g., "ROCO_00084.jpg") to captions.
    """
    image_data = {}
    try:
        with open(test_metadata_json_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                image_name = data["file_name"]
                # Extract only the filename if it's a path
                image_name = os.path.basename(image_name)
                image_data[image_name] = data["text"]
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {test_metadata_json_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {test_metadata_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred while reading metadata: {e}")
    return image_data


class ClipScore():
    """
    Calculates CLIP scores for generated images based on provided captions.
    It expects generated images to be in subfolders named after the first few words
    of the caption.
    """
    def __init__(self, args, image_caption_pairs, generated_image_filename="fig_1.png"):
        """
        Initializes the ClipScore calculator.

        Args:
            args: Configuration object with attributes:
                  - output_path (str): Base path for the output JSONL file.
                  - image_folders_path (str): Root directory of generated image folders.
            captions (list): A list of captions to score against images.
        """
        print("\n","="*60,"Initializing CLIP Score Calculation","="*60)
        self.model, self.preprocess = load("ViT-B/32", device=device)
        self.results = []
        self.output_path = f"{args.output_path}_clip.jsonl"
        self.image_caption_pairs=image_caption_pairs
        self.image_folders_path = args.image_folders_path
        self.generated_image_filename=generated_image_filename
        self.run_clip()

    def run_clip(self):
        """
        Orchestrates the CLIP score calculation and output.
        """
        print("\n","="*60,"Calculating CLIP Score","="*60)
        self.get_clip_score()
        self.output_result()

    def calculate_clip_score(self, image_path, caption):
        """
        Calculates the CLIP similarity score between a single image and a caption.

        Args:
            image_path (str): Path to the image file.
            caption (str): The text caption.

        Returns:
            float: The cosine similarity score. Returns 0.0 if an error occurs.
        """
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            text_tokens = tokenize([caption]).to(device) # Changed variable name for clarity
            with torch.no_grad():   # Get features from the model
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text_tokens)
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
            return similarity
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path} for CLIP score.")
            return 0.0
        except Exception as e:
            print(f"Error calculating CLIP score for {image_path} and caption '{caption}': {e}")
            return 0.0
    
    def get_clip_score(self):
        """
        Matches images with captions based on a folder naming convention and calculates CLIP scores.
        The folder name is derived from the first three words of the caption.
        It picks the first image found in the corresponding folder.
        """
        for img_id, caption in self.image_caption_pairs:
            folder_name = img_id.split(".")[0]
            image_path = os.path.join(self.image_folders_path, folder_name,self.generated_image_filename)
            score = self.calculate_clip_score(image_path, caption)
            self.results.append((image_path, caption, score))
            print(f"Image: {image_path}, Caption: \"{caption}\", CLIP Score: {score:.4f}")

    def output_result(self):
        """
        Saves the calculated CLIP scores and the average score to a JSONL file.
        """
        if not self.results:
            print("No CLIP scores were calculated to output.")
            return

        clip_scores_values = [score for _, _, score in self.results] 
        
        output_dir = os.path.dirname(self.output_path)
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True) 
        
        avg_clip = np.mean(np.array(clip_scores_values)) if clip_scores_values else 0.0
        print(f"\nImage Folders Path (for CLIP): {self.image_folders_path}")
        print(f"Average CLIP Score: {avg_clip:.4f}")
        
        # with open(self.output_path, "w", encoding="utf-8") as f: 
        #     json.dump({"average_clip_score": round(avg_clip, 4), "image_folders_path": self.image_folders_path}, f, ensure_ascii=False)
        #     f.write("\n") 
        # print(f"CLIP Score results saved to {self.output_path}")


class FIDScore():
    """
    Calculates the Fréchet Inception Distance (FID) between a set of real images
    and a set of generated images.
    It processes images one by one, saving them to temporary folders for FID calculation.
    """
    def __init__(self, args, image_ls, generated_image_filename="fig_1.png", id_extension_to_replace=".jpg"):
        """
        Initializes the FIDScore calculator.

        Args:
            args: Configuration object with attributes:
                  - test_image_path (str): Root directory of real test images.
                  - image_folders_path (str): Root directory of generated image folders.
                  - output_path (str): Base path for the output JSONL file.
            image_ls (list): List of image identifiers (e.g., "ROCO_00084.jpg") used to
                             match real and generated images.
            generated_image_filename (str): The fixed filename of the generated image within its subfolder.
            id_extension_to_replace (str): The extension in image_ls identifiers to remove for subfolder naming.
        """
        print("\n", "="*60, "Initializing FID Score Calculation", "="*60)
        self.image_ls = image_ls
        self.real_image_root_path = os.path.join(args.test_image_path, "images") 
        self.gen_image_root_path = args.image_folders_path 
        self.output_path = args.output_path + "_fid.jsonl"
        self.results = [] 
        self.generated_image_filename = generated_image_filename
        self.id_extension_to_replace = id_extension_to_replace
        
        self.tmp_real_dir = "./tmp_real_for_fid"
        self.tmp_fake_dir = "./tmp_fake_for_fid"

        self.run_fid()

    def run_fid(self):
        """
        Orchestrates the FID score calculation and output.
        """
        print("\n", "="*60, "Calculating FID Score", "="*60)
        self.get_fid_scores()
        self.output_result()
        self._cleanup_temp_dirs()

    def calculate_fid_score_for_pair(self, real_img_temp_dir, gen_img_temp_dir):
        """
        Calculates FID score given paths to temporary directories containing one real and one fake image.
        """
        fid_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[real_img_temp_dir, gen_img_temp_dir],
            batch_size=1, 
            device=fid_device,
            dims=2048
        )
        return fid_value

    def get_fid_scores(self):
        """
        Iterates through image_ls, prepares image pairs in temporary directories,
        and calculates FID for each pair.
        """
        transform_to_save = transforms.Compose([ 
            transforms.Resize((299, 299)), 
        ])
        
        os.makedirs(self.tmp_real_dir, exist_ok=True)
        os.makedirs(self.tmp_fake_dir, exist_ok=True)

        for image_name_identifier in self.image_ls:
            real_img_full_path = os.path.join(self.real_image_root_path, image_name_identifier)
            
            image_id_base = image_name_identifier.replace(self.id_extension_to_replace, "")
            gen_img_subfolder_path = os.path.join(self.gen_image_root_path, image_id_base)
            gen_img_full_path = os.path.join(gen_img_subfolder_path, self.generated_image_filename)

            temp_real_img_path = os.path.join(self.tmp_real_dir, "current_real_img.png") 
            temp_fake_img_path = os.path.join(self.tmp_fake_dir, "current_fake_img.png") 

            try:
                if not os.path.exists(real_img_full_path):
                    print(f"Warning: Real image not found at {real_img_full_path}. Skipping FID for {image_name_identifier}.")
                    continue
                real_img_pil = Image.open(real_img_full_path).convert("RGB")
                real_img_transformed_pil = transform_to_save(real_img_pil)
                real_img_transformed_pil.save(temp_real_img_path)

                if not os.path.exists(gen_img_full_path):
                    print(f"Warning: Generated image not found at {gen_img_full_path}. Skipping FID for {image_name_identifier}.")
                    continue
                gen_img_pil = Image.open(gen_img_full_path).convert("RGB")
                gen_img_transformed_pil = transform_to_save(gen_img_pil)
                gen_img_transformed_pil.save(temp_fake_img_path)
                
                fid = self.calculate_fid_score_for_pair(self.tmp_real_dir, self.tmp_fake_dir)
                self.results.append((image_name_identifier, fid))
                print(f"{image_name_identifier} → FID: {fid:.4f}")

            except FileNotFoundError as e:
                print(f"Error: Image file not found during FID processing for {image_name_identifier}: {e}")
                continue
            except Exception as e:
                print(f"Error processing images for FID ({image_name_identifier}): {e}")
                continue
            finally:
                if os.path.exists(temp_real_img_path):
                    os.remove(temp_real_img_path)
                if os.path.exists(temp_fake_img_path):
                    os.remove(temp_fake_img_path)

    def output_result(self):
        """
        Saves individual FID scores and the average FID score to a JSONL file.
        """
        if not self.results:
            print("No FID scores were calculated to output.")
            return

        fids_values = [score for _, score in self.results]
        avg_fid = np.mean(fids_values) if fids_values else 0.0
        
        print(f"\nAverage FID Score: {avg_fid:.4f}")
        
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # with open(self.output_path, "w", encoding="utf-8") as f:
        #     for image_name, fid in self.results:
        #         record = {
        #             "image_name": image_name,
        #             "fid_score": round(fid, 4) 
        #         }
        #         # f.write(json.dumps(record) + "\n")
            
        #     avg_record = {
        #         "average_fid_score": round(avg_fid, 4),
        #         "real_images_path": self.real_image_root_path, 
        #         "generated_images_path": self.gen_image_root_path
        #     }
        #     f.write(json.dumps(avg_record) + "\n")
        # print(f"FID Score results saved to {self.output_path}")

    def _cleanup_temp_dirs(self):
        """Cleans up the temporary directories created for FID calculation."""
        try:
            import shutil
            if os.path.exists(self.tmp_real_dir):
                shutil.rmtree(self.tmp_real_dir)
            if os.path.exists(self.tmp_fake_dir):
                shutil.rmtree(self.tmp_fake_dir)
        except Exception as e:
            print(f"Error cleaning up temporary FID directories: {e}")


class InceptionScore:
    """
    Calculates the Inception Score for a set of generated images.
    """
    def __init__(self, args, image_ls, batch_size=32, n_splits=10, generated_image_filename="fig_1.png", id_extension_to_replace=".jpg"):
        """
        Initializes the InceptionScore calculator.
        """
        print("\n", "="*60, "Initializing Inception Score Calculation", "="*60)
        self.args = args
        self.image_ls = image_ls
        self.gen_image_path_root = args.image_folders_path 
        self.output_path = args.output_path + "_inception.jsonl" 
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.generated_image_filename = generated_image_filename
        self.id_extension_to_replace = id_extension_to_replace
        self.results_summary = {} 
        self.device = device 
        print(f"Using device for Inception Score: {self.device}")
        self.inception_model = inception_v3(weights='Inception_V3_Weights.DEFAULT', transform_input=False)
        self.inception_model.to(self.device)
        self.inception_model.eval() 
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.run_score_calculation()

    def run_score_calculation(self):
        """
        Main orchestrator for calculating and outputting the Inception Score.
        """
        print("\n", "="*60, "Calculating Inception Score", "="*60)
        self._calculate_aggregate_inception_score()
        self._output_results()

    def _get_predictions_for_all_images(self):
        """
        Loads all specified generated images, preprocesses them, and gets
        softmax predictions from the Inception V3 model.
        """
        all_preds_list = []
        num_processed_images = 0
        temp_image_batch_tensors = [] 
        print(f"Processing {len(self.image_ls)} image identifiers for Inception Score...")
        for i, image_identifier in enumerate(self.image_ls):
            image_id_base = image_identifier.replace(self.id_extension_to_replace, "")
            gen_img_subfolder_path = os.path.join(self.gen_image_path_root, image_id_base)
            gen_img_full_path = os.path.join(gen_img_subfolder_path, self.generated_image_filename)
            if i > 0 and i % 100 == 0 : 
                print(f"  Processed {i}/{len(self.image_ls)} identifiers for Inception Score...")
            try:
                if not os.path.exists(gen_img_full_path):
                    continue
                img_pil = Image.open(gen_img_full_path).convert("RGB")
                processed_img_tensor = self.transform(img_pil) 
                temp_image_batch_tensors.append(processed_img_tensor)
                num_processed_images += 1
                if len(temp_image_batch_tensors) == self.batch_size or \
                   (i == len(self.image_ls) - 1 and temp_image_batch_tensors):
                    batch_tensor = torch.stack(temp_image_batch_tensors).to(self.device)
                    with torch.no_grad():
                        preds_logits = self.inception_model(batch_tensor)
                        softmax_preds = F.softmax(preds_logits, dim=1)
                    all_preds_list.append(softmax_preds.cpu()) 
                    temp_image_batch_tensors = [] 
            except FileNotFoundError:
                print(f"Warning (IS): Image file not found at {gen_img_full_path}. Skipping.")
            except Exception as e:
                print(f"Error loading or processing image {gen_img_full_path} for Inception Score: {e}")
                continue
        print(f"Finished image loading for Inception Score. Total images successfully processed: {num_processed_images}")
        if not all_preds_list:
            print("Warning (IS): No images were successfully processed or no predictions generated.")
            return None, 0
        all_preds_tensor = torch.cat(all_preds_list, dim=0)
        return all_preds_tensor, num_processed_images

    def _calculate_inception_score_from_predictions(self, preds_tensor, eps=1e-16):
        """
        Calculates Inception Score from a tensor of softmax predictions.
        """
        N = preds_tensor.size(0)
        if N == 0:
            print("Warning (IS): No predictions available to calculate Inception Score.")
            return 0.0, 0.0
        actual_n_splits = self.n_splits
        if N < self.n_splits:
            print(f"Warning (IS): Number of images ({N}) is less than n_splits ({self.n_splits}). "
                  f"Using {N} splits if N > 0, or 1 split if N is very small.")
            actual_n_splits = max(1, N) 
        if N < 10 and self.n_splits > N : 
             print(f"Warning (IS): Very few images ({N}) for {self.n_splits} splits. IS might be unstable. Consider more images.")
        split_scores = []
        imgs_per_split = N // actual_n_splits 
        if imgs_per_split == 0 and N > 0: 
            print(f"Warning (IS): Number of images per split is 0 (N={N}, splits={actual_n_splits}). Using all images for 1 split.")
            imgs_per_split = N
            actual_n_splits = 1
        elif imgs_per_split == 0 and N == 0:
            return 0.0, 0.0
        for k in range(actual_n_splits):
            start_idx = k * imgs_per_split
            end_idx = start_idx + imgs_per_split
            if k == actual_n_splits - 1:
                end_idx = N 
            part = preds_tensor[start_idx:end_idx, :]
            if part.size(0) == 0: 
                continue
            p_y = torch.mean(part, dim=0) 
            log_p_yx = torch.log(part + eps)
            log_p_y = torch.log(p_y + eps).unsqueeze(0) 
            kl_div_per_image = torch.sum(part * (log_p_yx - log_p_y), dim=1) 
            mean_kl_for_split = torch.mean(kl_div_per_image)
            split_scores.append(torch.exp(mean_kl_for_split))
        if not split_scores:
            print("Warning (IS): Could not calculate Inception Score for any split.")
            return 0.0, 0.0
        mean_is = torch.mean(torch.stack(split_scores)).item()
        std_is = torch.std(torch.stack(split_scores)).item() if len(split_scores) > 1 else 0.0
        return mean_is, std_is

    def _calculate_aggregate_inception_score(self):
        """
        Gathers all generated image predictions and computes the overall Inception Score.
        """
        all_preds_tensor, num_processed = self._get_predictions_for_all_images()
        if all_preds_tensor is None or num_processed == 0:
            print("Inception Score calculation aborted due to lack of processed images.")
            self.results_summary = { 
                "mean_inception_score": 0.0,
                "std_inception_score": 0.0,
                "num_images_processed": 0
            }
            return
        print(f"Calculating Inception Score from {num_processed} images' predictions using {self.n_splits} configured splits...")
        mean_is, std_is = self._calculate_inception_score_from_predictions(all_preds_tensor)
        self.results_summary = { 
            "mean_inception_score": mean_is,
            "std_inception_score": std_is,
            "num_images_processed": num_processed
        }
        print(f"\nProcessed {num_processed} images for Inception Score.")
        print(f"Mean Inception Score: {mean_is:.4f}")
        print(f"Std Dev Inception Score: {std_is:.4f}")

    def _output_results(self):
        """
        Saves the calculated Inception Score (mean and std) to a JSONL file.
        """
        if not self.results_summary: 
            print("No Inception Score results to output.")
            return
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        record = {
            "mean_inception_score": round(self.results_summary.get("mean_inception_score", 0.0), 4),
            "std_inception_score": round(self.results_summary.get("std_inception_score", 0.0), 4),
            "num_images_processed": self.results_summary.get("num_images_processed", 0),
            "generated_images_root_path": self.gen_image_path_root,
            "image_list_size_input": len(self.image_ls), 
            "batch_size_used": self.batch_size,
            "n_splits_configured": self.n_splits
        }
        # try:
        #     with open(self.output_path, "w", encoding="utf-8") as f:
        #         f.write(json.dumps(record) + "\n")
        #     print(f"\nInception Score results saved to: {self.output_path}")
        # except IOError as e:
        #     print(f"Error writing Inception Score results to {self.output_path}: {e}")


class LPIPSScore:
    """
    Calculates the Learned Perceptual Image Patch Similarity (LPIPS) between
    pairs of real and generated images.
    """
    def __init__(self, args, image_ls, generated_image_filename="fig_1.png", 
                 id_extension_to_replace=".jpg", lpips_net='alex'):
        """
        Initializes the LPIPSScore calculator.

        Args:
            args: Configuration object.
            image_ls (list): List of image identifiers.
            generated_image_filename (str): Filename of generated images.
            id_extension_to_replace (str): Extension to replace in identifiers.
            lpips_net (str): Network type for LPIPS ('alex' or 'vgg').
        """
        print("\n", "="*60, "Initializing LPIPS Score Calculation", "="*60)
        self.image_ls = image_ls
        self.real_image_root_path = os.path.join(args.test_image_path, "images")
        self.gen_image_root_path = args.image_folders_path
        self.output_path = args.output_path + "_lpips.jsonl"
        self.results = []  # Stores (image_name, lpips_score)
        self.generated_image_filename = generated_image_filename
        self.id_extension_to_replace = id_extension_to_replace
        self.device = device
        self.lpips_net_name = lpips_net

        # Load LPIPS model
        try:
            self.lpips_model = lpips.LPIPS(net=lpips_net).to(self.device)
            self.lpips_model.eval() # Set to evaluation mode
            print(f"LPIPS model ({lpips_net}) loaded on {self.device}.")
        except Exception as e:
            print(f"Error loading LPIPS model: {e}. LPIPS calculation will be skipped.")
            self.lpips_model = None # Ensure it's None if loading fails
            return # Exit init if model fails to load

        self.run_lpips()

    def _preprocess_image_for_lpips(self, image_path):
        """
        Loads an image, converts it to a PyTorch tensor, and normalizes it to [-1, 1].
        LPIPS expects BCHW format.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            # No resize by default, LPIPS can handle various sizes, but often images are resized
            # to a common size like 256x256 before LPIPS if consistency is desired.
            # For now, we use original size as transformed by ToTensor.
            transform_lpips = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(), # Converts to [0, 1] range, CHW
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalizes to [-1, 1] range
            ])
            img_tensor = transform_lpips(img).unsqueeze(0).to(self.device) # Add batch dim: BCHW
            return img_tensor
        except FileNotFoundError:
            print(f"LPIPS Preprocessing: Image not found at {image_path}")
            return None
        except Exception as e:
            print(f"LPIPS Preprocessing: Error processing image {image_path}: {e}")
            return None

    def run_lpips(self):
        """
        Orchestrates LPIPS calculation and output.
        """
        if not self.lpips_model: # Check if model loaded successfully
            print("LPIPS model not available. Skipping LPIPS calculation.")
            return

        print("\n", "="*60, "Calculating LPIPS Score", "="*60)
        self.get_lpips_scores()
        self.output_result()

    def get_lpips_scores(self):
        """
        Iterates through image_ls, loads real and generated image pairs,
        preprocesses them, and calculates LPIPS distance.
        """
        if not self.lpips_model:
            return

        for image_identifier in self.image_ls:
            real_img_full_path = os.path.join(self.real_image_root_path, image_identifier)
            
            image_id_base = image_identifier.replace(self.id_extension_to_replace, "")
            gen_img_subfolder_path = os.path.join(self.gen_image_root_path, image_id_base)
            gen_img_full_path = os.path.join(gen_img_subfolder_path, self.generated_image_filename)

            # Preprocess images
            real_img_tensor = self._preprocess_image_for_lpips(real_img_full_path)
            gen_img_tensor = self._preprocess_image_for_lpips(gen_img_full_path)

            if real_img_tensor is None or gen_img_tensor is None:
                print(f"Skipping LPIPS for {image_identifier} due to image loading/processing errors.")
                continue

            try:
                with torch.no_grad():
                    distance = self.lpips_model(real_img_tensor, gen_img_tensor).item()
                self.results.append((image_identifier, distance))
                print(f"{image_identifier} (Real) vs {os.path.basename(gen_img_full_path)} (Gen) → LPIPS: {distance:.4f}")
            except Exception as e:
                print(f"Error calculating LPIPS for {image_identifier}: {e}")
                continue
    
    def output_result(self):
        """
        Saves individual LPIPS scores and the average LPIPS score to a JSONL file.
        """
        if not self.results:
            print("No LPIPS scores were calculated to output.")
            return

        lpips_scores_values = [score for _, score in self.results]
        avg_lpips = np.mean(lpips_scores_values) if lpips_scores_values else 0.0
        
        print(f"\nAverage LPIPS Score: {avg_lpips:.4f} (Lower is better)")
        
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # with open(self.output_path, "w", encoding="utf-8") as f:
        #     for image_name, score in self.results:
        #         record = {
        #             "image_name": image_name,
        #             "lpips_score": round(score, 4)
        #         }
        #         # f.write(json.dumps(record) + "\n")
            
        #     avg_record = {
        #         "average_lpips_score": round(avg_lpips, 4),
        #         "real_images_path": self.real_image_root_path, 
        #         "generated_images_path": self.gen_image_root_path,
        #         "lpips_net": self.lpips_net_name if self.lpips_model else "N/A"
        #     }
        #     f.write(json.dumps(avg_record) + "\n")
        # print(f"LPIPS Score results saved to {self.output_path}")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluation script for image generation models.")
    parser.add_argument("--output_path",type=str, default="./save_score/default_run", help="Base path for saving score files (e.g., ./save_score/run_name). Suffixes like _clip.jsonl will be added.")
    parser.add_argument("--image_folders_path",type=str, required=True, help="Path to the root directory of generated image folders.")
    parser.add_argument("--test_image_path",type=str, required=True, help="Path to the root directory of the test image dataset (e.g., ../dataset/test). Expected to contain 'metadata.jsonl' and an 'images/' subdirectory for FID/LPIPS.")
    
    parser.add_argument("--clip_score", action='store_true', help="Calculate CLIP Score.")
    parser.add_argument("--fid_score", action='store_true', help="Calculate FID Score.")
    parser.add_argument("--inception_score", action='store_true', help="Calculate Inception Score.")
    parser.add_argument("--lpips_score", action='store_true', help="Calculate LPIPS Score.")

    # Optional arguments for Inception Score
    parser.add_argument("--is_batch_size", type=int, default=32, help="Batch size for Inception Score calculation.")
    parser.add_argument("--is_n_splits", type=int, default=10, help="Number of splits for Inception Score calculation.")
    
    # Optional arguments for LPIPS Score
    parser.add_argument("--lpips_net", type=str, default='alex', choices=['alex', 'vgg'], help="Network backbone for LPIPS (alex or vgg). Default: alex.")

    # Common arguments for image identification (used by FID, Inception Score, LPIPS)
    parser.add_argument("--generated_image_filename", type=str, default="fig_1.png", help="Filename of the generated image within its subfolder (e.g., image_id/fig_1.png).")
    parser.add_argument("--id_extension_to_replace", type=str, default=".jpg", help="Extension to remove from image list identifiers to get subfolder names (e.g., .jpg from ROCO_00084.jpg).")

    args = parser.parse_args()
    return args

def save_combined_scores(args, avg_scores):
    """
    Saves all average scores to a single JSON file.

    Args:
        args: Configuration object with output_path attribute.
        avg_scores (dict): Dictionary containing average scores for each metric.
    """
    combined_output_path = f"{args.output_path}_scores.json"
    output_dir = os.path.dirname(combined_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(combined_output_path, "w", encoding="utf-8") as f:
            json.dump(avg_scores, f, ensure_ascii=False, indent=4)
        print(f"Combined average scores saved to {combined_output_path}")
    except Exception as e:
        print(f"Error saving combined scores to {combined_output_path}: {e}")

def main():
    args = parse_args()
    
    print("\nScript Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 60)

    if args.output_path:
        base_output_dir = os.path.dirname(args.output_path)
        if base_output_dir and not os.path.exists(base_output_dir): 
            os.makedirs(base_output_dir, exist_ok=True)
            print(f"Created output directory: {base_output_dir}")
    
    avg_scores = {}  # Dictionary to store average scores
    image_caption_pairs = []
    image_identifiers = [ 
        "ROCO_00084.jpg", "ROCO_00138.jpg", "ROCO_00264.jpg", "ROCO_00350.jpg",
        "ROCO_00454.jpg", "ROCO_00509.jpg", "ROCO_00599.jpg", "ROCO_00611.jpg",
        "ROCO_00723.jpg", "ROCO_00747.jpg", "ROCO_02754.jpg", "ROCO_03231.jpg"
    ]
    
    metadata_path = os.path.join(args.test_image_path, "metadata.jsonl")
    image_to_caption_map = get_image_caption(metadata_path)

    if not image_to_caption_map:
        print("Warning: Could not load image-to-caption map. CLIP score might be affected.")
    
    for img_id in image_identifiers:
        if img_id in image_to_caption_map:
            image_caption_pairs.append((img_id, image_to_caption_map[img_id]))
        else:
            print(f"Warning: Caption not found for image ID '{img_id}' in metadata. It will be skipped for CLIP if it was intended.")

    if args.clip_score:
        if image_caption_pairs:
            clip_calculator = ClipScore(
                args=args,
                image_caption_pairs=image_caption_pairs,
                generated_image_filename=args.generated_image_filename
            )
            # Extract average CLIP score from results
            clip_scores_values = [score for _, _, score in clip_calculator.results]
            avg_clip = np.mean(np.array(clip_scores_values)) if clip_scores_values else 0.0
            avg_scores["average_clip_score"] = round(avg_clip, 4)
        else:
            print("Skipping CLIP score calculation as no captions were prepared.")
            avg_scores["average_clip_score"] = 0.0

    if args.fid_score:
        fid_calculator = FIDScore(
            args=args,
            image_ls=image_identifiers,
            generated_image_filename=args.generated_image_filename,
            id_extension_to_replace=args.id_extension_to_replace
        )
        # Extract average FID score from results
        fids_values = [score for _, score in fid_calculator.results]
        avg_fid = np.mean(fids_values) if fids_values else 0.0
        avg_scores["average_fid_score"] = round(avg_fid, 4)

    if args.inception_score:
        inception_calculator = InceptionScore(
            args=args,
            image_ls=image_identifiers,
            batch_size=args.is_batch_size,
            n_splits=args.is_n_splits,
            generated_image_filename=args.generated_image_filename,
            id_extension_to_replace=args.id_extension_to_replace
        )
        # Extract mean Inception Score from results_summary
        avg_scores["average_inception_score"] = round(
            inception_calculator.results_summary.get("mean_inception_score", 0.0), 4
        )
        avg_scores["inception_std"] = round(
            inception_calculator.results_summary.get("std_inception_score", 0.0), 4
        )

    if args.lpips_score:
        lpips_calculator = LPIPSScore(
            args=args,
            image_ls=image_identifiers,
            generated_image_filename=args.generated_image_filename,
            id_extension_to_replace=args.id_extension_to_replace,
            lpips_net=args.lpips_net
        )
        # Extract average LPIPS score from results
        lpips_scores_values = [score for _, score in lpips_calculator.results]
        avg_lpips = np.mean(lpips_scores_values) if lpips_scores_values else 0.0
        avg_scores["average_lpips_score"] = round(avg_lpips, 4)

    # Save all average scores to a single JSON file
    if avg_scores:
        save_combined_scores(args, avg_scores)
    else:
        print("No scores were calculated to save in combined output.")

    print("\n" + "="*60 + " Evaluation Script Finished " + "="*60)

if __name__ == "__main__":
    main()