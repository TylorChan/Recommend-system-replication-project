input_file = "requirements.txt"
output_file = "requirements_mac.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        # Skip comments and Conda platform lines
        if line.startswith("#") or "platform:" in line or "=" not in line:
            continue
        pkg = line.split("=")[0].strip()
        ver = line.split("=")[1].strip()
        # Convert to pip format
        fout.write(f"{pkg}=={ver}\n")
