# Game of Life

The default pattern is Tim Coe's Spacefiller 2, a quadratic-growth pattern that expands into zebra stripes. Build and upload it from `proj/` with:

```sh
make -C sw/game_of_life
make upload game_of_life
```

Patterns are standard RLE files under `assets/`. Select one by its filename without the `.rle` suffix:

```sh
make -C sw/game_of_life PATTERN=gosper_glider_gun
```

Included choices are `spacefiller2`, `gosper_glider_gun`, `acorn`, `r_pentomino`, `p80_gliderless_hwss`, `p156_eight_barrel`, `space_rake`, `pufferfish`, `breeder1`, and `simkin_glider_gun`.

The new patterns can be selected in the same way, for example:

```sh
make -C sw/game_of_life PATTERN=p80_gliderless_hwss
make -C sw/game_of_life PATTERN=p156_eight_barrel
make -C sw/game_of_life PATTERN=space_rake
make -C sw/game_of_life PATTERN=pufferfish
make -C sw/game_of_life PATTERN=breeder1
make -C sw/game_of_life PATTERN=simkin_glider_gun
```

Changing `PATTERN` automatically repacks and relinks the asset. To add another pattern, place its standard B3/S23 RLE file in `assets/`; `main.c` reads the packed dimensions and centers it automatically.
