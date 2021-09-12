
def cmn_get_all_image_path_from_folder(folder_path, include_sub_folder=False):
    import os
    if not os.path.exists(folder_path):
        print(f"for getting img , path not exists , err_path: {folder_path}")
    else:
        img_ext_to_pull = (".jpg", ".jpeg",".png",".tiff")
        img_files_paths = []
        # this for recurse main + all sub dir
        for root, Dirs, files in os.walk(folder_path):
            print(f"r : {root}, Dirs : {Dirs}, files : {files}")
            if files:
                for file in files:
                    for ext in img_ext_to_pull:
                        if str(file).__contains__(ext):
                            file_path = os.path.join(root, file)
                            img_files_paths.append(file_path)
            if not include_sub_folder:
                break
        
        return img_files_paths