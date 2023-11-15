from pathlib import Path


def write_sld_file(
    values: list[float | int], labels: list[str], colors: list[str], output_file: Path
) -> None:
    breakpoint()
    entries = [
        f'    <ColorMapEntry color="{e[0]}" quantity="{e[1]}" label="{e[2]}" />'
        for e in reversed(list(zip(colors, values, labels)))
    ]
    entries.insert(0, '  <ColorMap type="ramp" extended="false">')
    entries.insert(0, "<RasterSymbolizer>")
    entries.append("  </ColorMap>")
    entries.append("</RasterSymbolizer>")
    with open(output_file, "w") as dst:
        for entry in entries:
            dst.write(entry + "\n")


if __name__ == "__main__":
    wofs_colors = [
        "#ffffff",
        "#cc2100",
        "#e3e309",
        "#00e330",
        "#00ddca",
        "#1d0ae3",
        "#1d0ae3",
    ]

    wofs_labels = [
        "Water not detected",
        "Water detected in 1% of observations (includes flooding and misclassified shadows",
        "Water detected in 5% of observations (includes intermittant water bodies",
        "Water detected in 20% of observations (includes water bodies that often dry out",
        "Water detected in 50% of observations",
        "Water detected in 80% of observations (permanent water bodies)",
        "Water detected always",
    ]
    wofs_labels = [""] * len(wofs_colors)

    wofs_values = [0, 1, 30, 50, 60, 97.5, 100]

    write_sld_file(wofs_values, wofs_labels, wofs_colors, "wofs_sld.xml")
