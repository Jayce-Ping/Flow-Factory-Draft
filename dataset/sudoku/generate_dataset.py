"""
Sudoku Dataset Generator
"""
import json
import random
import copy
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class SudokuProcessor:
    def __init__(self, size=9, img_size=512, font_scale=0.6):
        self.size = size
        self.img_size = img_size
        self.cell_size = img_size / size
        self.font_scale = font_scale
        self.grid = [[0] * size for _ in range(size)]
        self.solution = None
        self.ocr = None

    # ==================== Generation ====================
    
    def _is_valid(self, grid, row, col, num):
        if num in grid[row]:
            return False
        if num in [grid[i][col] for i in range(self.size)]:
            return False
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if grid[br + i][bc + j] == num:
                    return False
        return True
    
    def _solve(self, grid):
        for i in range(self.size):
            for j in range(self.size):
                if grid[i][j] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if self._is_valid(grid, i, j, num):
                            grid[i][j] = num
                            if self._solve(grid):
                                return True
                            grid[i][j] = 0
                    return False
        return True
    
    def _count_solutions(self, grid, limit=2):
        grid = copy.deepcopy(grid)
        count = [0]
        
        def backtrack():
            if count[0] >= limit:
                return
            for i in range(self.size):
                for j in range(self.size):
                    if grid[i][j] == 0:
                        for num in range(1, 10):
                            if self._is_valid(grid, i, j, num):
                                grid[i][j] = num
                                backtrack()
                                grid[i][j] = 0
                        return
            count[0] += 1
        
        backtrack()
        return count[0]
    
    def find_all_solutions(self, puzzle, limit=100):
        if isinstance(puzzle, str):
            puzzle = self.decode(puzzle)
        
        solutions = []
        grid = copy.deepcopy(puzzle)
        
        def backtrack():
            if len(solutions) >= limit:
                return
            for i in range(9):
                for j in range(9):
                    if grid[i][j] == 0:
                        for num in range(1, 10):
                            if self._is_valid(grid, i, j, num):
                                grid[i][j] = num
                                backtrack()
                                grid[i][j] = 0
                        return
            solutions.append(copy.deepcopy(grid))
        
        backtrack()
        return solutions
    
    def generate(self, clues=40):
        self.grid = [[0] * self.size for _ in range(self.size)]
        self._solve(self.grid)
        self.solution = copy.deepcopy(self.grid)
        
        cells = list(range(81))
        random.shuffle(cells)
        to_remove = 81 - clues
        removed = 0
        
        for idx in cells:
            if removed >= to_remove:
                break
            r, c = divmod(idx, 9)
            if self.grid[r][c] == 0:
                continue
            
            backup = self.grid[r][c]
            self.grid[r][c] = 0
            
            if self._count_solutions(self.grid) != 1:
                self.grid[r][c] = backup
            else:
                removed += 1
        
        if removed < to_remove:
            return self.generate(clues)
        
        return self.grid, self.solution

    # ==================== Rendering ====================
    
    def render(self, grid, path, font_path=None, font_scale=None):
        img = Image.new("RGB", (self.img_size, self.img_size), "white")
        draw = ImageDraw.Draw(img)
        cs = self.cell_size
        
        scale = font_scale if font_scale is not None else self.font_scale
        font_size = int(cs * scale)
        
        try:
            font_path = font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        
        for i in range(10):
            w = 3 if i % 3 == 0 else 1
            draw.line([(i * cs, 0), (i * cs, self.img_size)], fill="black", width=w)
            draw.line([(0, i * cs), (self.img_size, i * cs)], fill="black", width=w)
        
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0:
                    text = str(grid[i][j])
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x = j * cs + (cs - tw) / 2
                    y = i * cs + (cs - th) / 2 - bbox[1]
                    draw.text((x, y), text, fill="black", font=font)
        
        img.save(path)
        return img

    # ==================== OCR Parsing (Cell-by-Cell) ====================
    
    def _init_ocr(self):
        from paddleocr import PaddleOCR
        if self.ocr is None:
            self.ocr = PaddleOCR(use_angle_cls=False, lang='en')
    
    def _crop_cell(self, img, row, col, padding=2):
        """Crop single cell from image with optional padding inward."""
        cs = self.cell_size
        x1 = int(col * cs) + padding
        y1 = int(row * cs) + padding
        x2 = int((col + 1) * cs) - padding
        y2 = int((row + 1) * cs) - padding
        return img.crop((x1, y1, x2, y2))
    
    def _parse_cell(self, cell_img):
        """
        Parse single cell image.
        
        Returns:
            (value, error): value is 0-9, error is None or error message
        """
        cell_arr = np.array(cell_img)
        result = self.ocr.predict(cell_arr)
        
        # Empty result → empty cell
        if not result or not result[0]:
            return 0, None
        
        # Extract all digits from all detected texts
        digits = []
        for text in result[0]['rec_texts']:
            if text.isdigit() and text != '0':
                digits.append(int(text))
        
        if len(digits) == 0:
            return 0, None
        elif len(digits) == 1:
            return digits[0], None
        else:
            # Multiple digits → error
            return -1, f"multiple_digits:{digits}"
    
    def parse(self, image_path, return_errors=False):
        """
        Parse Sudoku image cell-by-cell in fingerprint order (row-major).
        
        Args:
            image_path: Path to sudoku image
            return_errors: If True, also return error dict
        
        Returns:
            grid (and optionally errors dict {(row,col): error_msg})
        """
        self._init_ocr()
        
        img = Image.open(image_path).convert('RGB')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        grid = [[0] * 9 for _ in range(9)]
        errors = {}
        
        # Parse in fingerprint order (row-major: 0-80)
        for idx in range(81):
            row, col = divmod(idx, 9)
            cell_img = self._crop_cell(img, row, col)
            value, error = self._parse_cell(cell_img)
            
            grid[row][col] = max(0, value)  # -1 → 0 for grid
            if error:
                errors[(row, col)] = error
        
        return (grid, errors) if return_errors else grid
    
    def encode(self, grid):
        return ''.join(str(cell) for row in grid for cell in row)
    
    def decode(self, fingerprint):
        return [[int(fingerprint[i * 9 + j]) for j in range(9)] for i in range(9)]

    # ==================== Verification ====================
    
    def is_valid_solution(self, grid):
        full = set(range(1, 10))
        for row in grid:
            if set(row) != full:
                return False
        for col in range(9):
            if {grid[row][col] for row in range(9)} != full:
                return False
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                box = {grid[br + i][bc + j] for i in range(3) for j in range(3)}
                if box != full:
                    return False
        return True
    
    def _is_compatible(self, grid, puzzle):
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] != 0 and grid[i][j] != puzzle[i][j]:
                    return False
        return True
    
    def evaluate(self, parsed, ground_truth=None, puzzle=None, parse_errors=None):
        """
        Evaluate parsed result against ground truth(s).
        
        Args:
            parsed: Parsed grid or 81-char string
            ground_truth: Single or list of correct solutions
            puzzle: Original puzzle (required if ground_truth not provided)
            parse_errors: Dict of parse errors from parse(return_errors=True)
        
        Returns:
            dict with accuracy metrics
        """
        if isinstance(parsed, str):
            parsed = self.decode(parsed)
        if puzzle is not None and isinstance(puzzle, str):
            puzzle = self.decode(puzzle)
        
        if ground_truth is None:
            if puzzle is None:
                raise ValueError("Either ground_truth or puzzle must be provided")
            gts = self.find_all_solutions(puzzle)
            if not gts:
                raise ValueError("No valid solution found for the puzzle")
        else:
            if not isinstance(ground_truth, list):
                ground_truth = [ground_truth]
            gts = [self.decode(gt) if isinstance(gt, str) else gt for gt in ground_truth]
        
        best_result = None
        best_correct = -1
        
        for gt in gts:
            total = correct = 0
            given_total = given_correct = 0
            errors = []
            
            for i in range(9):
                for j in range(9):
                    p_val, gt_val = parsed[i][j], gt[i][j]
                    
                    if puzzle is not None:
                        if puzzle[i][j] == 0:
                            total += 1
                            if p_val == gt_val:
                                correct += 1
                            else:
                                errors.append((i, j, p_val, gt_val))
                        else:
                            given_total += 1
                            if p_val == gt_val:
                                given_correct += 1
                    else:
                        total += 1
                        if p_val == gt_val:
                            correct += 1
                        else:
                            errors.append((i, j, p_val, gt_val))
            
            if correct > best_correct:
                best_correct = correct
                best_result = {
                    "gt": gt,
                    "correct": correct,
                    "total": total,
                    "errors": errors,
                    "given_correct": given_correct,
                    "given_total": given_total,
                }
        
        is_valid = self.is_valid_solution(parsed)
        is_compatible = self._is_compatible(parsed, puzzle) if puzzle else True
        is_any_match = any(parsed == gt for gt in gts)
        
        total = best_result["total"]
        result = {
            "is_valid": is_valid,
            "is_compatible": is_compatible,
            "is_correct": is_any_match,
            "is_acceptable": is_valid and is_compatible,
            "accuracy": best_result["correct"] / total if total else 1.0,
            "correct": best_result["correct"],
            "total": total,
            "errors": best_result["errors"],
            "num_ground_truths": len(gts),
        }
        
        if puzzle is not None:
            gt = best_result["given_total"]
            result["given_accuracy"] = best_result["given_correct"] / gt if gt else 1.0
            result["given_correct"] = best_result["given_correct"]
            result["given_total"] = gt
        
        # Include parse errors if provided
        if parse_errors:
            result["parse_errors"] = parse_errors
        
        return result


def generate_dataset(
    output_dir: str = "sudoku",
    clue_levels: list = [30, 40, 50, 60, 70, 75],
    num_per_clue: int = 100,
    train_ratio: float = 0.8,
    prompt: str = "Solve this Sudoku puzzle using red font.",
    seed: int = 42
):
    """
    Generate deduplicated Sudoku dataset with train/test split.
    
    Args:
        output_dir: Root output directory
        clue_levels: List of clue counts to generate
        num_per_clue: Target puzzles per clue level
        train_ratio: Train split ratio
        prompt: Prompt text for JSONL
        seed: Random seed
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    proc = SudokuProcessor()
    seen_grids = set()  # Deduplication by fingerprint
    all_samples = []
    
    for clue in clue_levels:
        generated = 0
        attempts = 0
        max_attempts = num_per_clue * 10  # Prevent infinite loop
        
        while generated < num_per_clue and attempts < max_attempts:
            attempts += 1
            puzzle, solution = proc.generate(clue)
            fingerprint = proc.encode(puzzle)
            
            if fingerprint in seen_grids:
                continue
            
            seen_grids.add(fingerprint)
            
            # Save image
            img_name = f"clue{clue}_{generated:04d}.png"
            img_path = img_dir / img_name
            proc.render(puzzle, img_path)
            
            all_samples.append({
                "prompt": prompt,
                "image": f"{img_name}",
                "clue": clue,
                "puzzle": fingerprint,
                "solution": proc.encode(solution)
            })
            generated += 1
        
        print(f"Clue {clue}: {generated} puzzles generated")
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    
    # # Write JSONL (only prompt + image for final output)
    # def write_jsonl(samples, path):
    #     with open(path, 'w') as f:
    #         for s in samples:
    #             json.dump({"prompt": s["prompt"], "image": s["image"]}, f)
    #             f.write('\n')
    
    # write_jsonl(train_samples, output_dir / "train.jsonl")
    # write_jsonl(test_samples, output_dir / "test.jsonl")
    
    # Also save full metadata for evaluation
    def write_full_jsonl(samples, path):
        with open(path, 'w') as f:
            for s in samples:
                json.dump(s, f)
                f.write('\n')
    
    write_full_jsonl(train_samples, output_dir / "train.jsonl")
    write_full_jsonl(test_samples, output_dir / "test.jsonl")
    
    print(f"\nDataset saved to {output_dir}/")
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")
    print(f"  Total unique puzzles: {len(all_samples)}")


if __name__ == "__main__":
    generate_dataset(
        output_dir='./',
        num_per_clue=200,
    )