import tkinter as tk
import tkinter.font as tkfont
import cv2
import mediapipe as mp
import numpy as np
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import random
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from ttkthemes import ThemedStyle
from tkinter import Tk, Label

window = tk.Tk()
window.resizable(width=False, height=False)
window.geometry('650x700+5+5')  # 전체화면으로 설정
window.title("모두를 위한 수화 학습")

window.configure(background='white') # 메인 화면 배경색 설정

# 메인 화면 이미지 불러오기
image_path = "C:/models/Image/hand.jpg"
image = Image.open(image_path)

# 이미지를 표시하기 위한 레이블을 생성
image_label = tk.Label(window)
image_tk = ImageTk.PhotoImage(image) 
image_label.config(image=image_tk)
image_label.grid()
image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)# 메인 이미지 위치 지정

style = ThemedStyle(window)

style.set_theme("adapta")  # 원하는 테마 설정

# 종료 버튼 클릭 시 프로그램 종료
#def exit_program():
#    window.destroy()

model = load_model("C:/models/model.h5") # h5 모델 경로
sentence_model = load_model("C:/models/sentence_model.h5")
actions = ["나", "너" , "이름" , "집" , "있다" , "없다", "산" , "약속"]

seq_length = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands (max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
sentence_seq = []

### 동영상 경로 넣기 ###
hello_video = [
    "C:/models/hands/인사/나,저.mp4","C:/models/hands/인사/너,당신.mp4", "C:/models/hands/인사/반갑다.mp4", "C:/models/hands/인사/안녕하세요.mp4", "C:/models/hands/인사/이름.mp4", "C:/models/hands/인사/인사.mp4"
]

place_video = [
    "C:/models/hands/장소/고향.mp4", "C:/models/hands/장소/길.mp4", "C:/models/hands/장소/네,맞다.mp4", "C:/models/hands/장소/아파트.mp4","C:/models/hands/장소/이웃.mp4", "C:/models/hands/장소/집.mp4"
]

family_video = [
    "C:/models/hands/가족/딸.mp4", "C:/models/hands/가족/아들.mp4", "C:/models/hands/가족/아버지.mp4", "C:/models/hands/가족/어머니.mp4", "C:/models/hands/가족/언니,누나.mp4", "C:/models/hands/가족/형,오빠.mp4"
]

time_video = [
    "C:/models/hands/시간/내일.mp4", "C:/models/hands/시간/어제.mp4", "C:/models/hands/시간/언제.mp4", "C:/models/hands/시간/없다.mp4", "C:/models/hands/시간/오늘.mp4", "C:/models/hands/시간/있다.mp4"
]

weather_video = [
    "C:/models/hands/날씨/계절.mp4", "C:/models/hands/날씨/눈.mp4", "C:/models/hands/날씨/다니다.mp4", "C:/models/hands/날씨/산.mp4", "C:/models/hands/날씨/싫다.mp4", "C:/models/hands/날씨/우산.mp4"
]

invite_video = [
    "C:/models/hands/초대/궁금하다.mp4", "C:/models/hands/초대/배고프다.mp4", "C:/models/hands/초대/배부르다.mp4","C:/models/hands/초대/약속.mp4", "C:/models/hands/초대/질문하다.mp4", "C:/models/hands/초대/초대하다.mp4"
]

######################################################################################################################################################################################

#학습 페이지
def study_page():
    # 단어 버튼 창 생성
    word_window = tk.Toplevel(window)
    word_window.title("학습 페이지")
    word_window.resizable(width=False, height=False)
    word_window.geometry('650x700+5+5')
    word_window.configure(bg="orange")  # 배경색 설정

    font = tkfont.Font(family="Pretendard-Bold", size=30, slant="roman", weight="bold") # 상단의 라벨 적기

    label = tk.Label(word_window,text = 'Study!', font=font,bg='orange') # top 쪽에 올라감
    label.place(x=265,y=35)

    # 왼쪽 프레임 만들기 (동영상 추출 공간) (650 * 628)    
    l_frame = tk.Frame(word_window, width= 650, height=628, bg='orange') # 왼쪽프레임
    l_frame.pack(side=tk.LEFT)

    # 프레임 위에 동영상 넣기
    canvas = tk.Canvas(l_frame, width=650, height=600, bg='orange')
    canvas.pack(side=tk.TOP)

    # 적절한 크기로 비디오 크기 조정
    def resize_video(frame, target_width, target_height):
        height, width, _ = frame.shape
        ratio = min(target_width/width, target_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame
    
    # 비디오 프레임 표시 함수 수정
    global current_video  # 전역 변수로 선언
    current_video = None  # 초기화

    def show_frame(video_path):
        global current_video

        # 이전 동영상 정지
        if current_video is not None:
            current_video.release()

        cap = cv2.VideoCapture(video_path)
        current_video = cap

        def update_frame():
            _, frame = cap.read()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = resize_video(frame, 600, 400) # 비디오 크기
                img = Image.fromarray(resized_frame)
                img = ImageTk.PhotoImage(img)
                canvas.create_image(25, 0, anchor=tk.NW, image=img) # X위치, Y위치
                canvas.image = img

            if cap.isOpened():
                l_frame.after(15, update_frame)
            else:
                cap.release()

        # 프레임 업데이트
        update_frame()
    
    word = ["인사", "장소", "가족", "시간", "날씨", "초대"] # 콤보 박스에 표시할 카테고리 목록
    combobox = ttk.Combobox(l_frame) # l_frame이라는 창에 콤보 박스 생성
    combobox.config(height=25) # 높이 설정
    combobox.config(width=52) # 너비 설정
    combobox.config(justify="center") # 콤보 박스 내용 가운데 정렬
    combobox.config(values=word) # 표시할 항목 목록 설정
    combobox.config(state="readonly") # 사용자가 직접 콤보 박스에 입력할 수 없음
    combobox.set("학습할 단어를 선택하세요") # 처음에 표시될 값 설정
    combobox.place(relx=0.5, rely=0.6, anchor=tk.CENTER) # 콤보 박스 위치 조정
    combobox.config(font=("Pretendard-Bold", 12))
    
    # 카태고리를 눌렀을 시 나오는 버튼 생성을 위한 함수
    def show_wordbuttons():
        selected_category = combobox.get()
        if selected_category == "인사":
            phrases = ["나/저", "너/당신", "반갑다", "안녕하세요", "이름", "인사"]
            paths = np.array(hello_video) # 동영상 경로
            font = ("Pretendard-Bold", 10)

        elif selected_category == "장소":
            phrases = ["고향", "길", "네/맞다", "아파트", "이웃", "집"]
            paths = np.array(place_video) # 동영상 경로
            font = ("Pretendard-Bold", 10)

        elif selected_category == "가족":
            phrases = ["딸", "아들", "아버지", "어머니", "언니/누나", "오빠/형"]
            paths = np.array(family_video) # 동영상 경로
            font = ("Pretendard-Bold", 10)

        elif selected_category == "시간":
            phrases = ["내일", "어제", "언제", "없다", "오늘", "있다" ]
            paths = np.array(time_video) # 동영상 경로
            font = ("Pretendard-Bold", 10)

        elif selected_category == "날씨":
            phrases = ["계절", "눈", "다니다", "산", "싫다", "우산"]
            paths = np.array(weather_video) # 동영상 경로
            font = ("Pretendard-Bold", 10)

        elif selected_category == "초대":
            phrases = ["궁금하다", "배고프다", "배부르다", "약속", "질문하다", "초대하다"]
            paths = np.array(invite_video) # 동영상 경로
            font = ("Pretendard-Bold", 10)

        else:
            return

        button_frame = ttk.Frame(l_frame)
        button_frame.place(relx=0.5, rely=0.8, anchor=tk.CENTER) # 카테고리 버튼 위치 조정

        row_count = 0
        column_count = 0

        for i in range(len(phrases)):
            button = tk.Button(button_frame, text=phrases[i], command=lambda path=paths[i]: show_frame(path), width=16, height=3, fg = "white" ,bg="#464646", font=("Pretendard-Bold", 15))
            button.grid(row=row_count, column=column_count)
            print(f"{phrases[i]} 버튼에 {paths[i]}이(가) 입력되었습니다.")
            column_count += 1
            if column_count == 3:
                column_count = 0
                row_count += 1

    # 콤보 박스에서 카테고리가 선택되었을 때 `show_wordbuttons()` 함수 호출
    combobox.bind("<<ComboboxSelected>>", lambda event: show_wordbuttons())

####################################################################################################################################################################################

actions = ["나", "너", "이름", "집", "있다", "없다", "산", "약속"]  # 수어 퀴즈 동작

# 이전에 선택된 수어 데이터를 기록하는 변수
previous_action = ""

def quiz_page():
    quiz_window = tk.Toplevel(window)  # 새로운 창 생성
    quiz_window.title("퀴즈 페이지")
    quiz_window.geometry("650x620+5+5")  # 창크기
    quiz_window.resizable(width=False, height=False) # 창크기 조절 비활성
    quiz_window.configure(bg="orange")  # 배경색 설정

    font = tkfont.Font(family="Pretendard-Bold", size=20, slant="roman", weight="bold")
    result_label_font = tkfont.Font(family="맑은 고딕", size=10, slant="roman", weight="bold")

    # 창을 항상 맨 위에 떠있도록 설정
    quiz_window.attributes('-topmost', True)

    label = tk.Label(quiz_window, text="Quiz!", font=font)
    label.pack()
    label.configure(bg="orange")  # 배경색 설정

    # 영상 프레임 표시를 위한 캔버스 생성
    canvas = tk.Canvas(quiz_window, width=620, height=390)
    canvas.pack()

    # 결과 값을 표시할 라벨(Label) 생성
    result_label = tk.Label(quiz_window, text="인식중...", width=10, height=5, fg="black", font=result_label_font)
    result_label.pack(side=tk.RIGHT)  # 라벨을 오른쪽에 배치
    result_label.configure(bg="orange")  # 배경색 설정

    # 랜덤으로 수어 데이터 선택
    global previous_action
    random_action = random.choice(actions)

    # 이전에 선택된 수어와 다른 수어를 선택할 때까지 반복해서 선택
    while random_action == previous_action:
        random_action = random.choice(actions)

    # 선택된 수어를 이전 수어로 기록
    previous_action = random_action

    # 수어 데이터를 텍스트로 보여주는 라벨(Label) 생성
    action_label = tk.Label(quiz_window, text="문제 : "+random_action, width=10, height=5, fg="black", font=font)
    action_label.pack(side=tk.LEFT)  # 라벨을 왼쪽에 배치
    action_label.configure(bg="orange")  # 배경색 설정

    def skip_action():
        # 스킵 버튼을 눌렀을 때 호출되는 함수
        custom_dialog = tk.Toplevel(quiz_window)
        answer = messagebox.askyesno(
            master=custom_dialog,
            title="알림",
            message="문제를 스킵하시겠습니까?"
        )
        if answer:
            # 다음 문제로 이동
            quiz_window.destroy()
            quiz_page()

    # 스킵 버튼 생성
    skip_button = ttk.Button(
        quiz_window,
        text="SKIP",
        command=skip_action,
        width=4
    )
    skip_button.place(x=578, y=1)  # 좌표 지정하여 배치
    
    # OpenCV에서 가져온 이미지를 Tkinter에서 표시하기 위한 함수
    def show_frame():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1) #좌우반전
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(frame)

        if result.multi_hand_landmarks is not None: #조건문을 사용하여 손이 감지되었는지 확인
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4)) #21개의 랜드마크를 저장하기 위한 2차원 배열을 생성
                for j, lm in enumerate(res.landmark): #각 랜드마크에 대한 반복문 j는 랜드마크의 인덱스, lm은 랜드마크 객체
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                #동작 인식 결과 표시(카메라 외부)
                if (len(action_seq) >= 7 and action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == action_seq[-6] == action_seq[-7]):
                    result_text = action_seq[-1].upper()

                    if result_text == random_action:
                        custom_dialog = tk.Toplevel(quiz_window)
                        messagebox.showinfo(master=custom_dialog, title="정답!", message="정답입니다!\n다음 문제를 준비해주세요!")
                        quiz_window.destroy()  # quiz_window 종료
                        quiz_page()  # 기존의 quiz_page 다시 시작
                else:
                    result_text = "인식중..."

                # 관절 간의 각도를 계산
                
                v1 = joint[  # Parent joint
                    [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,], :3, ]  
                
                v2 = joint[ # Child joint
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,], :3, ]  
                
                v = v2 - v1  # [20, 3]

                # 벡터 v를 정규화
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 각도를 구하기
                angle = np.arccos(
                    np.einsum(
                        "nt,nt->n",
                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
                    )
                )

                angle = np.degrees(angle)  # radian을 degree로 변환

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims (np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 7:
                    continue

        # OpenCV 이미지를 PIL 이미지로 변환
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)

        # Tkinter 창에 이미지 표시
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

        quiz_window.after(15, show_frame)  # 15ms마다 프레임 업데이트

    show_frame()  # 비디오 프레임 표시 시작

def translate_page():
    translate_window = tk.Toplevel(window)  # 새로운 창 생성
    translate_window.title("문장 페이지")
    translate_window.geometry("650x620+5+5")  # 창크기
    translate_window.resizable(width=False, height=False) # 창크기 조절 비활성
    translate_window.configure(bg="orange")  # 배경색 설정

    font = tkfont.Font(family="Pretendard-Bold", size=20, slant="roman", weight="bold")
    result_label_font = tkfont.Font(family="맑은 고딕", size=10, slant="roman", weight="bold")

    # 창을 항상 맨 위에 떠있도록 설정
    translate_window.attributes('-topmost', True)

    label = tk.Label(translate_window, text="Translate!", font=font)
    label.pack()
    label.configure(bg="orange")  # 배경색 설정
    # 1초 후에 텍스트를 변경
    translate_window.after(3000, lambda: label.config(text="나의 집은 산에 있다."))

    # 영상 프레임 표시를 위한 캔버스 생성
    canvas = tk.Canvas(translate_window, width=620, height=400)
    canvas.pack()

    # 결과 값을 표시할 라벨(Label) 생성
    result_label = tk.Label(translate_window, text="인식중...", width=10, height=5, fg="black", font=result_label_font)
    result_label.pack(side=tk.RIGHT)  # 라벨을 오른쪽에 배치
    result_label.configure(bg="orange")  # 배경색 설정

    # 수어 데이터를 텍스트로 보여주는 라벨(Label) 생성
    action_label = tk.Label(translate_window, text="문장: ", width=20, height=10, fg="black", font=font)
    action_label.pack(side=tk.LEFT)  # 라벨을 왼쪽에 배치
    action_label.configure(bg="orange")  # 배경색 설정

    recognized_first = []
    recognized_second = []
    recognized_third = []
    recognized_fourth = []
    recognized_fifth = []
    def first_sentence():

        result_text = action_seq[-1].upper()
        executed_one = False
        
        if(result_text == "나"):
            recognized_first.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("나" in recognized_first and result_text == "집"):
            recognized_first.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("나" in recognized_first and "집" in recognized_first and result_text == "산"):
            recognized_first.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("나" in recognized_first and "집" in recognized_first and "산" in recognized_first and result_text == "있다"):
            recognized_first.append(result_text)
            print(result_text, "가 들어갔어요")

        required_words = ["나", "집", "산", "있다"]

        valid_word = all(word in recognized_first for word in required_words)

        if valid_word:
            action_label.config(text="문장: 나의 집은 산에 있다.")
            translate_window.after(2000, lambda: label.config(text="나는 약속이 없다."))
            executed_one = True  # executed_one 변수를 True로 설정하여 첫 번째 동작이 실행되었음을 표시
            
        else:
            translate_window.after(500, first_sentence)

        return executed_one  # 첫 번째 동작이 성공적으로 실행되었는지 여부를 반환
        
    
    def second_sentence():
        executed_two = False
        result_text = action_seq[-1].upper()

        if(result_text == "나"):
            recognized_second.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("나" in recognized_second and result_text == "약속"):
            recognized_second.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("나" in recognized_second and "약속" in recognized_second and result_text == "없다"):
            recognized_second.append(result_text)
            print(result_text, "가 들어갔어요")

        required_words = ["나", "약속", "없다"]

        valid_word = all(word in recognized_second for word in required_words)

        if valid_word:
           action_label.config(text="문장: 나는 약속이 없다.")
           translate_window.after(2000, lambda: label.config(text="다음에 봐"))
           executed_two = True

        else:
           translate_window.after(500, second_sentence)
           
        return executed_two
    
    def third_sentence():
        executed_three = False
        result_text = action_seq[-1].upper()
        #recognized_sentence_third = " ".join(action_seq)

        if(result_text == "다음"):
            recognized_third.append(result_text)
            print(result_text, "가 들어갔어요")
            action_label.config(text="문장: " + result_text)
        elif("다음" in recognized_third and result_text == "봐"):
            recognized_third.append(result_text)
            print(result_text, "가 들어갔어요")

        required_words = ["다음", "봐"]

        valid_word = all(word in recognized_third for word in required_words)

        if valid_word:
           action_label.config(text="문장: 다음에 봐")
           translate_window.after(2000, lambda: label.config(text="너랑 나는 산에 있다."))
           executed_three = True

        else:
            translate_window.after(500, third_sentence)
        
        return executed_three
    
    def fourth_sentence():
        executed_four = False
        result_text = action_seq[-1].upper()
        #recognized_sentence_third = " ".join(action_seq)

        if(result_text == "너"):
            recognized_fourth.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("너" in recognized_fourth and result_text == "나"):
            recognized_fourth.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("너" in recognized_fourth and "나" and result_text == "산"):
            recognized_fourth.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("너" in recognized_fourth and "나" in recognized_fourth and "산" and result_text == "있다"):
            recognized_fourth.append(result_text)
            print(result_text, "가 들어갔어요")

        required_words = ["너", "나", "산", "있다"]

        valid_word = all(word in recognized_fourth for word in required_words)

        if valid_word:
           action_label.config(text="문장: 너랑 나는 산에 있다.")
           translate_window.after(2000, lambda: label.config(text="너의 집에서 약속이 있다."))
           executed_four = True

        else:
            translate_window.after(500, fourth_sentence)
        
        return executed_four
    
    def fifth_sentence():
        executed_five = False
        result_text = action_seq[-1].upper()
        #recognized_sentence_third = " ".join(action_seq)

        if(result_text == "너"):
            recognized_fifth.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("너" in recognized_fifth and result_text == "집"):
            recognized_fifth.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("너" in recognized_fifth and "집" and result_text == "약속"):
            recognized_fifth.append(result_text)
            print(result_text, "가 들어갔어요")
        elif("너" in recognized_fifth and "집" in recognized_fifth and "약속" and result_text == "있다"):
            recognized_fifth.append(result_text)
            print(result_text, "가 들어갔어요")

        required_words = ["너", "집", "약속", "있다"]

        valid_word = all(word in recognized_fifth for word in required_words)

        if valid_word:
           action_label.config(text="너의 집에서 약속이 있다.")
           translate_window.after(2000, lambda: action_label.config(text="모든 문장이 끝났습니다."))
           translate_window.after(2000, lambda: label.config(text="모든 문장이 끝났습니다."))
           executed_five = True

        else:
            translate_window.after(500, fifth_sentence)
        
        return executed_five
    #1. 나는 이름이 있다
    #2. 내가 집에 있다
    # OpenCV에서 가져온 이미지를 Tkinter에서 표시하기 위한 함수

    def show_frame():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1) #좌우반전
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(frame)
        
        # 초기에는 first만 실행하도록 플래그를 설정합니다.
        executed_first = False
        executed_second = False
        executed_third = False
        executed_fourth = False
        executed_fifth = False


        if result.multi_hand_landmarks is not None: #조건문을 사용하여 손이 감지되었는지 확인
            if len(action_seq) >= 7 and action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == action_seq[-6] == action_seq[-7]:

                # result_text = action_seq[-1].upper()
                # action_label.config(text="문장: " + result_text)

                num_sentences = 5
                
                # executed_first, ~ third까지 if문으로 한 번 씩 실행하게 만듬 위 함수 3개 있음 돌아가는 로직 보고 종료 후 print(action_seq) 해보면 지금까지 했던 동작 모두 들어 있는데 이걸 초기화 해야함
                for _ in range(num_sentences):
                    
                    # 첫 번째 동작 실행
                    if not executed_first:
                        result_text = action_seq[-1].upper()
                        action_label.config(text="문장: " + result_text)

                        executed_one = first_sentence()
                        executed_first = executed_one                        
                        
                    # 두 번째 동작 실행 (첫 번째가 이미 실행된 경우)
                    elif executed_first and not executed_second:
                        result_text = action_seq[-1].upper()
                        action_label.config(text="문장: " + result_text)

                        executed_two = second_sentence()
                        executed_second = executed_two

                    # 세 번째 동작 실행 (두 번째가 이미 실행된 경우)
                    elif executed_second and not executed_third:
                        result_text = action_seq[-1].upper()
                        #action_label.config(text="문장: " + result_text) 원래 여기에 이거 해야하는데, 모델 바꿔서 해서 third_sentence() 함수에 이 코드 적어서 실행 
                        
                        executed_three = third_sentence()
                        executed_third = executed_three

                    # 세 번째 동작 실행 (두 번째가 이미 실행된 경우)
                    elif executed_third and not executed_fourth:
                        result_text = action_seq[-1].upper()
                        action_label.config(text="문장: " + result_text)
                        
                        executed_four = fourth_sentence()
                        executed_fourth = executed_four
                        
                    # 세 번째 동작 실행 (두 번째가 이미 실행된 경우)
                    elif executed_fourth and not executed_fifth:
                        result_text = action_seq[-1].upper()
                        action_label.config(text="문장: " + result_text)
                        
                        executed_five = fifth_sentence()
                        executed_fifth = executed_five  

            
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4)) #21개의 랜드마크를 저장하기 위한 2차원 배열을 생성
                for j, lm in enumerate(res.landmark): #각 랜드마크에 대한 반복문 j는 랜드마크의 인덱스, lm은 랜드마크 객체
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # 관절 간의 각도를 계산
                
                v1 = joint[  # Parent joint
                    [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19,], :3, ]  
                
                v2 = joint[ # Child joint
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,], :3, ]  
                
                v = v2 - v1  # [20, 3]

                # 벡터 v를 정규화
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 각도를 구하기
                angle = np.arccos(
                    np.einsum(
                        "nt,nt->n",
                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
                    )
                )

                angle = np.degrees(angle)  # radian을 degree로 변환

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims (np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                #sentence_model
                if executed_second and not executed_third:
                    sentence_actions = ["다음", "봐"]
                    y_pred = sentence_model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.9:
                        continue

                    action = sentence_actions[i_pred]
                    action_seq.append(action)
                else:
                    y_pred = model.predict(input_data).squeeze()

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.9:
                        continue

                    action = actions[i_pred]
                    action_seq.append(action)

                

                if len(action_seq) < 7:
                    continue

        # OpenCV 이미지를 PIL 이미지로 변환
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)

        # Tkinter 창에 이미지 표시
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

        translate_window.after(15, show_frame)  # 15ms마다 프레임 업데이트

    show_frame()  # 비디오 프레임 표시 시작
    

style = ttk.Style()
style.configure("Custom.TButton", background="white", foreground="black", font=("Pretendard-Bold", 14), padding=(10, 20), borderwidth=5)

# 학습 버튼 테마 적용
button1 = ttk.Button(window, text="학습", style="Custom.TButton", command=study_page)
button1.grid(row=1, column=0, sticky="nsew")

# 퀴즈 버튼 테마 적용
button2 = ttk.Button(window, text="퀴즈", style="Custom.TButton", command=quiz_page)
button2.grid(row=1, column=1, sticky="nsew")

# 번역 버튼 테마 적용
button3 = ttk.Button(window, text="번역", style="Custom.TButton", command=translate_page)
button3.grid(row=1, column=2, sticky="nsew")

# 추가적으로, 모바일 어플처럼 화면 크기에 반응하도록 설정
for i in range(3):
    window.grid_columnconfigure(i, weight=1)


window.grid_rowconfigure(0, weight=0)  
window.grid_rowconfigure(0, weight=2) 

window.grid_columnconfigure(0, weight=1)  
window.grid_columnconfigure(1, weight=1)
window.grid_columnconfigure(2, weight=1)

# 로딩 창
loading_window = tk.Toplevel(window)
loading_window.title("로딩 중...")
loading_window.geometry("500x370")
loading_window.option_add("Pretendard-Bold", "a")

# GIF 이미지 로드
image = Image.open("C:/models/Image/hi.gif")
gif = ImageTk.PhotoImage(image)

# 이미지 애니메이션을 위한 PIL 객체 생성
frames = []
try:
    while True:
        frames.append(image.copy())
        image.seek(len(frames))  # 다음 프레임으로 이동
except EOFError:
    pass

# Label 위젯에 GIF 이미지 표시
label = tk.Label(loading_window, image=gif)
label.pack(pady=10)

# 애니메이션 업데이트 함수
def update_animation(frame_index=0):
    frame = frames[frame_index]
    photo = ImageTk.PhotoImage(frame)

    label.config(image=photo)
    label.image = photo  # 참조 유지

    # 다음 프레임으로 업데이트
    loading_window.after(100, update_animation, (frame_index + 1) % len(frames))

# 초기 애니메이션 업데이트 호출
update_animation()

# 프로그레스 바 생성
progress_bar = ttk.Progressbar(loading_window, orient="horizontal", mode="determinate", length=250,maximum=45)
progress_bar.pack(pady=20)

# 프로그레스 바 애니메이션 함수
def animate_progress(current_value, max_value, interval=100):
    progress_bar["value"] = current_value
    if current_value < max_value:
        loading_window.after(interval, animate_progress, current_value + 1, max_value)

# 로딩 창을 5초 뒤에 숨기고 메인 창을 표시하는 함수
def show_main_window():
    loading_window.withdraw()
    window.deiconify()

# 메인 창 숨기기
window.withdraw()

# 로딩 창 표시 및 애니메이션 실행
loading_window.after(0, animate_progress, 0, 100)  # 0ms 뒤부터 애니메이션 시작
loading_window.after(1000, show_main_window)  # 5초 뒤에 메인 창 표시
#loading_window.after(5000, show_main_window)
window.mainloop()