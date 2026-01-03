"""
Description:
Note: Make sure that both the new and old module files are in same directory!

This script helps you sync your previous module works with current modules.
It takes 2 arguments, source_dir_name and destination_dir_name.
All the files which will be moved are specified in files_to_sync.txt as newline separated strings

Usage: python sync_previous_module.py <source_dir_name> <dest_dir_name>

Ex:  python sync_previous_module.py mle-module-0-sauravpanda24 mle-module-1-sauravpanda24
"""
import os
import shutil
import sys

if len(sys.argv) != 3:
    print('Invalid argument count! Please pass source directory and destination directory after the file name')
    sys.exit()

# Get the users path to evaluate the username and root directory
current_path = os.getcwd()
grandparent_path = '/'.join(current_path.split('/')[:-1])

print('Looking for modules in : ', grandparent_path)

# List of files which we want to move
f = open('files_to_sync.txt', 'r+')
files_to_move = f.read().splitlines()
f.close()

# get the source and destination from arguments
source = sys.argv[1]
dest = sys.argv[2]


def resolve_root(arg):
    # Try the common candidate: relative to grandparent_path
    candidate = os.path.normpath(os.path.join(grandparent_path, arg))
    if os.path.exists(candidate):
        return candidate
    # Try basename placed under grandparent_path
    candidate2 = os.path.normpath(os.path.join(grandparent_path, os.path.basename(arg)))
    if os.path.exists(candidate2):
        return candidate2
    # Try absolute path provided by user
    candidate3 = os.path.abspath(arg)
    if os.path.exists(candidate3):
        return candidate3
    # Fallback to the first candidate (may not exist)
    return candidate


source_root = resolve_root(source)
dest_root = resolve_root(dest)

print(f"Resolved source root: {source_root}")
print(f"Resolved dest root: {dest_root}")

# copy the files from source to destination
moved_count = 0
for file in files_to_move:
    print(f"Moving file : {file}")
    src = os.path.normpath(os.path.join(source_root, file))
    dst = os.path.normpath(os.path.join(dest_root, file))
    try:
        if not os.path.exists(src):
            print(f"Source not found: {src} — skipping")
            continue
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst)
        moved_count += 1
    except Exception as e:
        print(f"Failed to copy {file}: {e}")
print(f"Finished moving {moved_count} files")
