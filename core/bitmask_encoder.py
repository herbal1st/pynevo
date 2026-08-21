"""
PyBiwis 64-bit packed integer bitmask grid encoder and decoder.
"""

from typing import List


class BitmaskEncoder:
    """
    Packs 2D tile matrix grids into 64-bit PyBiwis integer chunks.
    """

    @staticmethod
    def encode_grid_to_chunks(
        grid: List[List[int]],
        width: int,
        height: int
    ) -> List[int]:
        """
        Packs 2D tile matrix grid into 64-bit integer chunk array.
        """
        total_tiles: int = width * height
        num_chunks: int = (total_tiles + 63) // 64
        chunks: List[int] = [0] * num_chunks

        for y in range(height):
            for x in range(width):
                if grid[y][x] == 1:
                    flat_idx: int = x + (y * width)
                    chunk_idx: int = flat_idx // 64
                    bit_off: int = flat_idx % 64
                    chunks[chunk_idx] |= (1 << bit_off)

        return chunks

    @staticmethod
    def decode_chunks_to_grid(
        chunks: List[int],
        grid: List[List[int]],
        width: int,
        height: int
    ) -> None:
        """
        Unpacks 64-bit PyBiwis integer chunk array into 2D tile matrix grid.
        """
        for y in range(height):
            for x in range(width):
                flat_idx: int = x + (y * width)
                chunk_idx: int = flat_idx // 64
                bit_off: int = flat_idx % 64
                is_set: bool = bool(
                    (chunks[chunk_idx] >> bit_off) & 1
                )
                grid[y][x] = 1 if is_set else 0
