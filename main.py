import streamlit as st
from google.colab import drive

# Kết nối với Drive
drive.mount("/content/drive")

# Chuẩn bị mô hình
model_path = "/content/drive/MyDrive/streamlit-models/trained_model.pth"
model = torch.load(model_path)
# Định nghĩa hàm xử lý ảnh
def process_image(image):
    # Chuyển đổi ảnh sang dạng tensor
    image = torchvision.transforms.ToTensor()(e)
    # Chuẩn hóa ảnh
    image = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    # Đoán chữ trong ảnh
    prediction = model(image).argmax(dim=1)
    return prediction

# Tạo ứng dụng Streamlit
st.title("In chữ trong hình")

# Cho phép người dùng tải lên ảnh
image = st.file_uploader("Chọn ảnh")

# Xử lý ảnh
if image is not None:
    image = image.read()
    prediction = process_image(image)

# Hiển thị kết quả
st.write("Chữ trong ảnh là:", prediction)
