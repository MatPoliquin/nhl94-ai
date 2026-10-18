# Current limitations

- Legacy button frame skipping stops on termination but can continue repeated
  steps on truncation. Target-position mode stops on either flag.
- The older PvP display can raise `NameError` while creating its surface. Normal
  single-player debug playback uses a separate view.
- Legacy custom MLP classification and self-play observation handling have
  restrictions. The legacy self-play opponent encoder uses its default input
  configuration; arbitrary observation layouts and temporal opponents are not
  interchangeable with it.
- Export accepts Box observations and compatible button policies. Dict
  observations and target-position exports are unsupported.
