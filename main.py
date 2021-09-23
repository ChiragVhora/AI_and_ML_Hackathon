# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import requests


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def check_numpy_list_comprehension():
    import numpy
    images = (1,2,3,4,5,6)
    labels = [1,1,1,0,0,0]
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    print(f"images type {type(images)} data : {images} , labels - type{type(labels)} data : {labels}")
    pass


def check_fisher_face_recognizer():
    import cv2
    model = cv2.face.FisherFaceRecognizer_create()
    print(model)
    pass


def make_list_from_list_of_tuples_for_prediction():
    prediction = [(x/10, 1-x/10) for x in range(10)]
    # print(prediction)
    true_pred_list = [true_pred for _,true_pred in prediction]
    print(true_pred_list)
    pass


def check_negative_slicing():
    list_ = [i for i in range(10)]
    slice_list = list_[:-2]
    print(f"list : {list_} \nSliced : {slice_list}")
    pass


def check_pandas():
    import pandas as pd
    # import kerastuner
    df = pd.DataFrame({'X': [78, 85, 96, 80, 86], 'Y': [84, 94, 89, 83, 86], 'Z': [86, 97, 96, 72, 83]});
    print(df)
    pass

def check_h_and_W():
    # import cv2
    # import imutils
    #
    # frame_width = 600
    # frame_height = 600
    #
    # camera_instance = cv2.VideoCapture(0)
    # img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # resized_original_img = imutils.resize(img_from_cam, width=frame_width, height=frame_height)  # 500
    # shape = resized_original_img.shape
    # frame_height, frame_width, _ = resized_original_img.shape
    # print(frame_height, frame_width, _)
    pass

# for downloading img functions
def cmd_progressBar(current, total, barLength = 50, message="Progress"):
    total -= 1
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    if current == 0:
        print(f"{message}: ")
    print(f' [{arrow}{spaces}] {int(percent)} %', end='\r')
    # print(f'Progress: [{arrow}{spaces}] {int(percent)} %', end='\n')    # multiple line

def download_images_of_(search_key, dir_save_to, images_count=1, img_type="jpg"):
    '''
        reference video : https://youtu.be/V_MV5EsdKRc
    '''

    import os
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait

    from selenium.webdriver.common.keys import Keys
    import time

    PATH = r"C:\Local Disk  (D)\ALL projects\chromedriver.exe"

    # chrome_option = Options()
    # chrome_option.add_argument("--headless")
    # driver = webdriver.Chrome(PATH, options=chrome_option)
    driver = webdriver.Chrome(PATH)

    url = "https://www.google.com/"
    driver.get(url)

    # wait for 1 seconds
    WebDriverWait(driver, 1)

    search_input_box = driver.find_element_by_xpath('/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')

    for c in search_key:
        search_input_box.send_keys(c)
        time.sleep(0.215)

    search_input_box.send_keys(Keys.ENTER)

    # go to image section
    images_option =  driver.find_element_by_xpath('//*[@id="hdtb-msb"]/div[1]/div/div[2]/a')
    images_option.click()

    # scroll through all images to load them
    last_height = driver.execute_script('return document.body.scrollHeight')
    j = 0
    while j < 5:
        driver.execute_script('scrollTo(0, document.body.scrollHeight)')
        time.sleep(2)
        new_height = driver.execute_script('return document.body.scrollHeight')
        if new_height == last_height:
            break
        last_height = new_height
        j += 1

    # links_elems = driver.find_element_by_class_name("rg_meta")
    # image_links = []
    # for elem in links_elems:
    #     image_links.append(elem.text)
    # print(image_links)
    # time.sleep(10)

    # find all images tag
    all_img_tags = driver.find_elements_by_tag_name("img")
    print(f"total {len(all_img_tags)} found for {search_key}")

    # for img in all_img_tags:
    #     print(dir(img))
    #     print(url)
    #     break

    urls = []
    src_found_count = 0
    imgs_ext = (".png", ".jpg", ".tiff", ".jpeg")
    for image in all_img_tags:
        # try:
        url = image.get_attribute('src')
        if url:
            if not url.find('https://'):
                urls.append(url)
                src_found_count += 1
            else:
                for ext in imgs_ext:
                    if str(url).endswith(ext):  # img found not redirection
                        urls.append(url)
                        break
        if src_found_count >= images_count:       # we got out limit files
            break

    print(f"source found for {search_key} : {src_found_count}/{len(all_img_tags)}")
        # except:
        #     try:
        #         url = image.get_attribute('src')
        #         if not url.find('https://'):
        #             urls.append(image['src'])
        #     except Exception as e:
        #         print(f'No found image sources.')
        #         print(e)

    print("Downloading started ....")
    count = 0
    folder_T_T_V = ""
    if urls:
        # print("progress : ")
        for url in urls:
            # print(int(count/images_count *100), end="\r")
            cmd_progressBar(current=count, total=images_count, barLength=50)  # for progress
            # time.sleep(0.1)  # for checking the progress
            try:
                res = requests.get(url, verify=True, stream=True)
                rawdata = res.raw.read()
                
                # dividing data into Train:Test:VAl -> 90:10:10 % 
                if count >= images_count*.9:    # if greater then 80% -> remaining 10% to validation
                    folder_T_T_V = "validation"
                elif count >= images_count*.8:    # if greater then 80% -> remaining 10% to teat
                    folder_T_T_V = "test"
                else:
                    folder_T_T_V = "train"

                # making T,T,V folder
                folder_T_T_V_path = os.path.join(dir_save_to, folder_T_T_V)
                if not os.path.exists(folder_T_T_V_path):    # if not exists then create else append
                    os.mkdir(folder_T_T_V_path)

                # checking for folders in T,T,V folder
                save_img_to_path = os.path.join(folder_T_T_V_path, search_key)
                if not os.path.exists(save_img_to_path):
                    os.mkdir(save_img_to_path)


                image_path_with_name = f"{save_img_to_path}/{search_key}_{count}.{img_type}"

                with open(image_path_with_name, 'wb') as img_f:
                    img_f.write(rawdata)
                    count += 1

            except Exception as e:
                print(f'count : {count} , Failed to write raw data for folder {folder_T_T_V} , url : {url}.')
                print(e)
        print("\nDownload complete #################################################")
        print(f"\nDownload complete for {search_key} in the folder path : {dir_save_to}")
    # iterating images and saving them to folder
    # image_path_with_name = dir_save_to
    # for i in range(1, images_count):
    #     try:
    #         image_from_google = driver.find_element_by_xpath(f'//*[@id="islrg"]/div[1]/div[{i}]/a[1]/div[1]/img')
    #         image_path_with_name = dir_save_to + f"/{search_key}_{i}.jpg"
    #         time.sleep(0.1)
    #         image_from_google.screenshot(image_path_with_name)
    #     except Exception:
    #         print(f"iteration/img {i} : exception for {search_key} is {Exception}")
    #         pass
    pass


def download_images_using_google_image_download(query, img_folder_path, image_count=10):
    from google_images_download import google_images_download

    response = google_images_download.googleimagesdownload()

    arguments = {
        "keywords":query,
        "format": "jpg",
        "limit": image_count,
        "print_urls": True,
        "size": "medium",
        "output_directory": img_folder_path
    }
    # try:
    paths = response.download(arguments)  # passing the arguments to the function
    print(paths)  # printing absolute paths of the downloaded images
    # except Exception:
    #     print("exception in download_images_using_google_image_download , ", Exception)
    pass


def download_images():
    dataset_path = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_13_HandGesture_classification\Handgesture_Dataset"
    # image_key_folders = ["thanos", "joker"]
    query = ["hand thumb", "hand index finger", "hand ring finger", "hand fist", "hand palm"]
    method = 1
    if method ==1:
        for name in query:
            download_images_of_(name, dataset_path, images_count=150)
    else:
        # query_string =f"{image_key_folders[0]}, {image_key_folders[1]}"     # thanos,joker
        query_string = "thanos, joker"
        download_images_using_google_image_download(query_string, img_folder_path=dataset_path, image_count=70)
    pass


def check_get_img_path():
    import os
    test_folder_path = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_12_Inage_classification_CNN\1_hackathon_Image_classification_using_cnn\Dataset\test"
    from common_python_files.get_image_path import cmn_get_all_image_path_from_folder
    image_files_path = cmn_get_all_image_path_from_folder(test_folder_path, include_sub_folder=True)
    print(image_files_path)
    pass


if __name__ == '__main__':
    # print_hi('PyCharm')


    # function for solv problem tried   ###########################3

    # check_numpy_list_comprehension()
    # check_fisher_face_recognizer()
    # make_list_from_list_of_tuples_for_prediction()
    # check_negative_slicing()
    # check_pandas()

    # download_images()
    # download_ima()
    # check_get_img_path()
    # print("hello")
    n = 1000000
    for i in range(n):
        cmd_progressBar(i, n, message="first")

        cmd_progressBar(i, n, message="second")