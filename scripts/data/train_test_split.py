import os, json, re
import random

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# packages should be of the form ("PackageName", "OtherPackageName/Subdirectory", ...)
def split_data(
    current_test_file=os.path.join(
        ROOT_PATH, "scripts", "data", "test", "test_set_no_inst.json"
    ),
    packages=(
        "Compfiles",
        "MIL",
        # "PFR",
        # "PrimeNumberTheoremAnd",
        # "Mathlib/Analysis",
    ),
    train_portion=0.9,
):
    train = []
    test = []
    print(packages)
    with open(current_test_file, "r") as f:
        old_test = json.load(f)
    for pkg in packages:
        pkg_name = pkg.split("/")[0]
        if pkg_name in old_test:
            test.extend([(pkg_name, k) for k in old_test[pkg_name].keys()])

    total_theorems = 0
    for pkg in packages:
        # Find the (possibly differently-cased) parent directory of the package
        if os.path.exists(
            os.path.join(ROOT_PATH, ".lake", "packages", pkg.split("/")[0].lower())
        ):
            pkg_parent = pkg.split("/")[0].lower()
        elif os.path.exists(
            os.path.join(ROOT_PATH, ".lake", "packages", pkg.split("/")[0])
        ):
            pkg_parent = pkg.split("/")[0]
        else:
            print("couldn't find the package: " + pkg)
            continue
        pkg_path = os.path.join(
            ROOT_PATH, ".lake", "packages", pkg_parent, *pkg.split("/")
        )
        if os.path.exists(pkg_path):
            for root, dirs, files in os.walk(pkg_path):
                for file in files:
                    if file.endswith(".lean"):
                        # package_name = ".".join(root.split(os.sep) + [file[:-5]]).split(
                        #     pkg.split("/")[0].lower() + "."
                        # )[-1]
                        # print(package_name)
                        package_name = os.path.relpath(os.path.join(root,file),os.path.dirname(pkg_path))
                        # print(package_name1)
                        # print(f"{pkg} : {package_name1}")
                        # print()
                        if (
                            pkg == "MIL"
                            and "solutions" not in package_name
                        ):
                            continue
                        if package_name in test:
                            continue
                        with open(os.path.join(root, file), "r") as f:
                            content = f.read()
                            # find all instances of "theorem", "lemma", etc.
                            thms = re.findall(
                                r"\b(lemma|theorem|corollary|example|proposition|remark|definition|axiom)\b",
                                content,
                            )
                            total_theorems += len(thms)
                        # print(package_name)
                        pkg_src = pkg.split('/')[0]
                        train.append((pkg_src, package_name))
    total_len = len(train) + len(test)
    random.shuffle(train)
    while len(test) < total_len * (1 - train_portion):
        test.append(train.pop())
    print(total_len, len(train), len(test))
    print("Total theorems found: " + str(total_theorems))

    # print(train)
    # print("\n\n\n\n\n\n============\n\n\n\n\n\n")
    # print(test)
    new_train = {}
    new_test = {}
    for k,v in train:
        if k in new_train.keys():
            new_train[k].append(v)
        else:
            new_train[k]=[v]
    for k,v in test:
        if k in new_test.keys():
            new_test[k].append(v)
        else:
            new_test[k]=[v]
            
    out = {"train": new_train, "test": new_test}
    with open(
        os.path.join(ROOT_PATH, "scripts", "data", "train_test_split.json"), "w"
    ) as f:
        json.dump(out, f)


if __name__ == "__main__":
    split_data()
