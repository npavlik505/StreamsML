import re

# Since we're only generating libstreams.pyf now
pyf_files = ["libstreams.pyf"]

for pyf_path in pyf_files:
    lines = open(pyf_path).readlines()

    new_lines = []
    in_interface = False
    current_sub = None

    for line in lines:
        if "interface" in line:
            in_interface = True
            new_lines.append(line)
            continue

        if "end interface" in line:
            in_interface = False
            new_lines.append(line)
            continue

        if in_interface and "subroutine" in line:
            match = re.search(r"subroutine\s+(\w+)", line)
            if match:
                current_sub = match.group(1)
                new_lines.append(line)
                if current_sub.startswith("wrap_"):
                    new_lines.append(f"        !f2py symbol: {current_sub}_\n")
                    new_lines.append(f"        !f2py name: {current_sub}\n")
                continue

        new_lines.append(line)

    with open(pyf_path, "w") as f:
        f.writelines(new_lines)

print("✅ Patched libstreams.pyf with !f2py symbol bindings.")

