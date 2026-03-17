cd ~/Documents/projects/shios_segmentations

for f in *.zip; do
    name="${f%.zip}"
    extracted=$(unzip -Z1 "$f" | head -1)
    unzip "$f"
    mv "$extracted" "${name}.xml"
done

mkdir -p ~/Documents/projects/masks_label_dict/segmentation-xmls
for f in *.zip; do unzip "$f" && mv *.xml ~/Documents/projects/masks_label_dict/segmentation-xmls; done