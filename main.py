import streamlit as st
import torchvision.models as models

# Chuẩn bị mô hình
model = models.resnet18(pretrained=True)

# Định nghĩa hàm xử lý ảnh
def process_image(image):
    # Chuyển đổi ảnh sang dạng tensor
    image = torchvision.transforms.ToTensor()(image)
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
