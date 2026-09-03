"""
PyBiwis 64-bit packed integer bitmask encoder for 16x16 chunk grids.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray

from world.tile_registry import TileRegistry


class ChunkBitmaskEncoder:
    """
    Packs and unpacks 16x16 tile matrices into 4 PyBiwis uint64 words.
    """

    CHUNK_SIZE: int = 16  # tiles
    TILES_PER_CHUNK: int = 256  # tiles
    WORD_SIZE: int = 64  # bits
    WORDS_PER_CHUNK: int = 4  # words

    @classmethod
    def encode_grid_to_words(
        cls,
        grid: NDArray[np.uint8],
        tile_registry: TileRegistry
    ) -> NDArray[np.uint64]:
        """
        Encodes a 16x16 uint8 tile grid into 4 uint64 PyBiwis bitmask words.
        """
        words: NDArray[np.uint64] = np.zeros(
            cls.WORDS_PER_CHUNK, dtype=np.uint64
        )

        for y in range(cls.CHUNK_SIZE):
            for x in range(cls.CHUNK_SIZE):
                tile_id: int = int(grid[y, x])
                if tile_registry.get_tile(tile_id).solid:
                    flat_idx: int = x + (y * cls.CHUNK_SIZE)
                    word_idx: int = flat_idx // cls.WORD_SIZE
                    bit_off: int = flat_idx % cls.WORD_SIZE
                    words[word_idx] |= np.uint64(1 << bit_off)

        return words

    @classmethod
    def decode_words_to_solid_mask(
        cls,
        words: NDArray[np.uint64]
    ) -> NDArray[np.bool_]:
        """
        Decodes 4 uint64 PyBiwis words back into a 16x16 boolean solid mask.
        """
        mask: NDArray[np.bool_] = np.zeros(
            (cls.CHUNK_SIZE, cls.CHUNK_SIZE), dtype=bool
        )

        for y in range(cls.CHUNK_SIZE):
            for x in range(cls.CHUNK_SIZE):
                flat_idx: int = x + (y * cls.CHUNK_SIZE)
                word_idx: int = flat_idx // cls.WORD_SIZE
                bit_off: int = flat_idx % cls.WORD_SIZE
                is_set: bool = bool(
                    (words[word_idx] >> np.uint64(bit_off)) & np.uint64(1)
                )
                mask[y, x] = is_set

        return mask
