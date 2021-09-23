def cmn_get_all_image_path_from_folder(source_folder, include_sub_folder=False, files_to_get_per_folder=None, get_total_img=None):
    import os
    if not os.path.exists(source_folder):
        print(f"for getting img , path not exists , err_path: {source_folder}")
    else:
        img_ext_to_pull = (".jpg", ".jpeg", ".png", ".tiff")
        imgs_path_to_return = []
        # this for recurse main + all sub dir
        for root, Dirs, files in os.walk(source_folder):
            print(f"r : {root}, Dirs : {Dirs}, files : {files}")
            img_files_paths_for_a_folder = []
            if files:
                for file in files:
                    for ext in img_ext_to_pull:
                        if str(file).lower().__contains__(ext):
                            file_path = os.path.join(root, file)
                            img_files_paths_for_a_folder.append(file_path)

            if files_to_get_per_folder:     # per folder it gets files =  len(files_to_get_per_folder)
                if len(img_files_paths_for_a_folder) > files_to_get_per_folder:  # if more img get then needed -> slice
                    img_files_paths_for_a_folder = img_files_paths_for_a_folder[:files_to_get_per_folder]
            imgs_path_to_return.extend(img_files_paths_for_a_folder)
            if not include_sub_folder:
                break
        if get_total_img:
            if len(imgs_path_to_return) > get_total_img:  # if more img get then needed -> slice
                imgs_path_to_return = get_total_img[:get_total_img]
        return imgs_path_to_return