import numpy as np
import random
import sys
import matplotlib.pyplot as plt

class MapGenerator:
    def __init__(self, height, width, seed=None):
        """
        :param height: 맵의 높이 (행 개수, Y)
        :param width: 맵의 너비 (열 개수, X)
        :param seed: 랜덤 시드 값
        """
        self.height = height
        self.width = width
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # 난이도별 설정
        self.stage_config = {
            1: {'types': ['rect'], 'overlap_prob': 0.3},
            2: {'types': ['rect', 'u_shape'], 'overlap_prob': 0.3},
            3: {'types': ['rect', 'u_shape', 'false_hole'], 'overlap_prob': 0.3},
            4: {'types': ['rect', 'u_shape', 'false_hole'], 'overlap_prob': 0.5},
        }

    def generate_map(self, stage=1, num_obstacles=None, return_metadata=False):
        """
        맵을 생성하여 반환합니다.
        :param stage: 난이도 (1~4)
        :param num_obstacles: 장애물 개수 (None이면 면적 기반 자동 설정)
        :param return_metadata: True일 경우 (grid, metadata) 튜플 반환
        :return: grid (2D numpy array) or (grid, metadata)
        """
        max_global_retries = 20
        
        for _ in range(max_global_retries):
            grid, metadata, success = self._try_generate_map(stage, num_obstacles)
            
            if success:
                # 유효성 검사 (Free Space 비율)
                free_ratio = np.sum(grid == 0) / (self.height * self.width)
                if free_ratio >= 0.5: # 최소 50% 이상 빈 공간
                    if return_metadata:
                        return grid, metadata
                    return grid
                else:
                    # 빈 공간 부족으로 인한 재생성
                    continue
        
        print("Warning: Failed to generate valid map within max retries. Returning last result.")
        if return_metadata:
            return grid, metadata
        return grid

    def _try_generate_map(self, stage, num_obstacles):
        grid = np.zeros((self.height, self.width), dtype=int)
        metadata = []
        
        config = self.stage_config.get(stage, self.stage_config[1])
        allowed_types = config['types']
        overlap_prob_threshold = config['overlap_prob']
        
        if num_obstacles is None:
            # 맵 크기 비례 자동 설정 (대략 면적의 5% 정도 개수?)
            # 장애물 크기가 다양하므로 개수는 적당히 설정
            area = self.height * self.width
            min_obs = max(3, int(area / 150)) # 1500칸 기준 10개
            max_obs = max(5, int(area / 60))  # 1500칸 기준 25개
            target_obs = self.rng.randint(min_obs, max_obs)
        else:
            target_obs = num_obstacles
            
        count = 0
        attempts = 0
        max_attempts = target_obs * 50 # 칸 채우기 실패 방지용 limit
        
        while count < target_obs and attempts < max_attempts:
            attempts += 1
            
            # 1. 장애물 타입 및 파라미터 선정
            o_type = self.rng.choice(allowed_types)
            
            # 크기 샘플링 (맵 크기의 10% ~ 25% 수준)
            min_len = 3
            max_h = max(min_len, int(self.height * 0.25))
            max_w = max(min_len, int(self.width * 0.25))
            
            h = self.rng.randint(min_len, max_h)
            w = self.rng.randint(min_len, max_w)
            
            thickness = 1
            if min(h, w) > 4:
                thickness = self.rng.randint(1, min(h, w) // 3)
            
            # 2. Shape 생성 (Local Coordinates)
            cells = []
            if o_type == 'rect':
                cells = self._get_rect_cells(h, w)
            elif o_type == 'u_shape':
                cells = self._get_u_shape_cells(h, w, thickness)
            elif o_type == 'false_hole':
                # 벽 어딘가에 구멍 뚫기 (최대 w-2t 크기)
                opening_size = self.rng.randint(1, max(1, w - 2*thickness))
                cells = self._get_u_shape_cells(h, w, thickness, back_opening=opening_size)
                
            # 3. 회전
            angle = self.rng.choice([0, 90, 180, 270])
            cells = self._rotate_cells(cells, angle)
            
            # 4. 위치 선정 (Margin 고려)
            # 셀들의 bounding box 계산
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)
            
            margin = 1 # 테두리 마진
            
            # (min_r + offset_r) >= margin  => offset_r >= margin - min_r
            # (max_r + offset_r) <= H-1-margin => offset_r <= H-1-margin - max_r
            
            r_min_bound = margin - min_r
            r_max_bound = self.height - 1 - margin - max_r
            c_min_bound = margin - min_c
            c_max_bound = self.width - 1 - margin - max_c
            
            if r_min_bound > r_max_bound or c_min_bound > c_max_bound:
                continue # 맵에 비해 장애물이 너무 큼
                
            offset_r = self.rng.randint(r_min_bound, r_max_bound)
            offset_c = self.rng.randint(c_min_bound, c_max_bound)
            
            abs_cells = [(r + offset_r, c + offset_c) for r, c in cells]
            
            # 5. Overlap 정책 확인
            intersection = 0
            for r, c in abs_cells:
                if grid[r, c] == 1:
                    intersection += 1
            
            overlap_ratio = intersection / len(abs_cells)
            
            # 겹침 모드 결정
            allow_overlap_mode = (self.rng.random() < overlap_prob_threshold) # 30% or 50%
            
            accept = False
            if intersection == 0:
                accept = True # 안 겹치면 무조건 통과 (Non-overlap 모드 포함)
            elif allow_overlap_mode:
                if overlap_ratio < 0.3: # 30% 미만으로만 겹칠 경우 허용
                    accept = True
            
            if accept:
                for r, c in abs_cells:
                    grid[r, c] = 1
                
                count += 1
                metadata.append({
                    'id': count,
                    'type': o_type,
                    'pos': (offset_r, offset_c),
                    'size': (h, w),
                    'angle': angle
                })
                
        return grid, metadata, True

    def _get_rect_cells(self, h, w):
        """
        직사각형: (0,0) ~ (h-1, w-1) 채움
        """
        cells = []
        for r in range(h):
            for c in range(w):
                cells.append((r, c))
        return cells

    def _get_u_shape_cells(self, h, w, t, back_opening=0):
        """
        U-shape (기본: 위쪽이 뚫린 형태)
        - Bottom Wall
        - Left Wall
        - Right Wall
        """
        cells = set()
        
        # 전체 박스 채우기
        for r in range(h):
            for c in range(w):
                cells.add((r, c))
        
        # 내부 비우기 (Top open)
        # 내부 영역: r [0 : h-t] (위쪽), c [t : w-t] (좌우 벽 사이)
        # Note: (row=0 is top)
        inner_r_min, inner_r_max = 0, h - t
        inner_c_min, inner_c_max = t, w - t
        
        if inner_r_max > inner_r_min and inner_c_max > inner_c_min:
            for r in range(inner_r_min, inner_r_max):
                for c in range(inner_c_min, inner_c_max):
                    if (r, c) in cells:
                        cells.remove((r, c))
        
        # False Hole 생성 (Back Wall = Bottom Wall 에 구멍)
        if back_opening > 0:
            # Bottom Wall 범위: r [h-t : h]
            # 중앙 기준으로 opening 크기만큼 제거
            center_c = w // 2
            half_open = max(1, back_opening // 2)
            c_start = max(0, center_c - half_open)
            c_end = min(w, center_c + half_open)
            
            for r in range(h - t, h):
                for c in range(c_start, c_end):
                    if (r, c) in cells:
                        cells.remove((r, c))
                        
        return list(cells)

    def _rotate_cells(self, cells, angle):
        """
        (0,0) 기준 회전이 아닌, 그냥 90도씩 좌표 변환
        90도: (r, c) -> (c, -r)
        """
        if angle == 0:
            return cells
        
        new_cells = []
        for r, c in cells:
            if angle == 90:
                new_cells.append((c, -r))
            elif angle == 180:
                new_cells.append((-r, -c))
            elif angle == 270:
                new_cells.append((-c, r))
        return new_cells

def visualize_map(grid):
    chars = {0: '.', 1: '#'}
    rows, cols = grid.shape
    print(f"Map Preview ({rows}x{cols}):")
    print("-" * (cols + 2))
    for r in range(rows):
        line = "|"
        for c in range(cols):
            line += chars[grid[r][c]]
        line += "|"
        print(line)
    print("-" * (cols + 2))

def plot_map_matplotlib(grid, title="Generated Map"):
    plt.figure(figsize=(10, 6))
    plt.imshow(grid, cmap='binary', origin='upper') # 0:white(free), 1:black(obstacle)
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    # Grid lines for better visualization
    plt.grid(which='major', color='gray', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    
    # Legend manually
    # Patch for obstacle
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='black', edgecolor='black', label='Obstacle'),
                       Patch(facecolor='white', edgecolor='black', label='Free Space')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test Code
    gen = MapGenerator(height=30, width=50, seed=42)
    
    print(">>> Generating Stage 3 Map (A+B+C) <<<")
    grid, meta = gen.generate_map(stage=3, return_metadata=True)
    
    visualize_map(grid)
    
    obs_count = len(meta)
    print(f"Total Obstacles Placed: {obs_count}")
    print("Metadata Sample (first 3):", meta[:3])

    # Free Space Check
    free_cells = np.sum(grid == 0)
    total_cells = grid.size
    print(f"Free Space: {free_cells}/{total_cells} ({free_cells/total_cells*100:.1f}%)")
    
    # Matplotlib Visualization
    print("Displaying map with matplotlib...")
    plot_map_matplotlib(grid, title=f"Generated Map (Stage 3, Seed 42) - {obs_count} Obstacles")
