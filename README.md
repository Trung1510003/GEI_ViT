Nhận dạng dáng đi (Gait Recognition) sử dụng Vision Transformer (ViT) và Gait Energy Image (GEI)

1. Tổng quan
- Dự án này tập trung vào nhận diện dáng đi, một kỹ thuật xác định danh tính sinh trắc học dựa trên mô hình chuyển động khi đi bộ của mỗi người. Sử dụng Vision Transformer (ViT) để phân tích các đặc trưng dáng đi được trích xuất từ chuỗi hình ảnh silhouette. Bộ dữ liệu chính được sử dụng là CASIA-B, chứa các chuỗi đi bộ từ nhiều góc nhìn trong các điều kiện: normal, bag, clothes

2. Bộ dữ liệu
- CASIA-B: Bộ dữ liệu gồm 124 đối tượng, 3 điều kiện đi bộ: mang túi, mặc áo, bình thường, và 11 góc nhìn từ 0° đến 180°. Mỗi chuỗi chứa các hình ảnh silhouette được sử dụng để tạo ảnh GEI cho việc huấn luyện mô hình

3. Phương pháp
   - Tiền xử lí dữ liệu: Tạo ảnh GEI bằng cách lấy trung bình các silhouette trong một chuỗi dáng đi
   - Mô hình: Sử dụng Vision Transformer để xử lí với đầu vào là các ảnh GEI
   - Hàm mất mát: CrossEntropyLoss
   - Phương pháp đánh giá: PCA
