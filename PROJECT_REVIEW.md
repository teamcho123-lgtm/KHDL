# Báo cáo review toàn bộ dự án

**Ngày review:** 17/11/2026  
**Phiên bản được rà soát:** commit `d9c302b` (`feat: implement Phase 5 FastAPI dashboard, demo samples, and physiological safeguard`)  
**Phạm vi:** API, pipeline inference, huấn luyện, xử lý dữ liệu, đặc trưng âm học, nhận dạng người nói, giao diện web, cấu hình, tài liệu và script kiểm tra.  
**Cách thực hiện:** review tĩnh mã nguồn và đối chiếu luồng giữa các module. Không chạy tests trong lượt review này.

## Kết luận chung

Dự án đã có một prototype tương đối đầy đủ: chuẩn bị manifest theo người nói, huấn luyện mô hình lai Wav2Vec2 và đặc trưng âm học, baseline cá nhân, FastAPI và dashboard. Luồng xử lý được chia module và các file tạm do API tạo được dọn trong `finally`.

Hiện dự án **chưa sẵn sàng để mở API ra mạng hoặc dùng kết quả như một phép đo stress đáng tin cậy**. Các vấn đề lớn nhất là API không có xác thực, giới hạn tài nguyên được kiểm tra sau khi multipart đã được đọc và audio đã giải mã, cùng logic có thể ép kết quả thành `CALM` khi một số phép đo âm học thực ra thất bại. Ngoài ra, nhãn huấn luyện là nhãn cảm xúc RAVDESS được dùng làm đại diện cho stress; điều đó chưa chứng minh mô hình đo stress ngoài đời.

### Mức độ ưu tiên

- **P1:** cần xử lý trước khi triển khai cho nhiều người dùng, mở API trên mạng hoặc dựa vào kết quả để ra quyết định.
- **P2:** cần xử lý trước khi gọi bản phát hành là ổn định hoặc tái lập được.
- **P3:** cải thiện trải nghiệm và khả năng bảo trì.

## Phát hiện

### P1 — API không xác thực, cho phép ghi đè hồ sơ baseline và liệt kê ID

**Vị trí:** [`api/main.py`](api/main.py#L43), [`api/main.py`](api/main.py#L99), [`api/main.py`](api/main.py#L147), [`src/pipeline.py`](src/pipeline.py#L222)

API bật CORS cho mọi origin. Các route đăng ký baseline, phân tích và liệt kê ID đều không yêu cầu xác thực. Đăng ký cùng `user_id` sẽ thay thế file profile hiện có bằng `os.replace`. Nếu service được bind ra LAN/Internet, người không được phép có thể dò ID, thay hồ sơ baseline hoặc tiêu thụ tài nguyên inference. README có nhắc chỉ bind loopback và cần đặt xác thực trước khi mở mạng, nhưng code không tự giới hạn việc triển khai như vậy.

**Đề xuất:** thêm xác thực/ủy quyền cho các route ghi và đọc dữ liệu người dùng; giới hạn CORS theo danh sách origin cấu hình; tránh trả danh sách ID toàn hệ thống; giới hạn quyền ghi đè và bổ sung thao tác xóa/đổi baseline có kiểm soát. Giữ bind loopback làm mặc định triển khai demo.

### P1 — Giới hạn upload chưa chặn được xử lý multipart và giải mã audio tốn tài nguyên

**Vị trí:** [`api/main.py`](api/main.py#L54), [`src/pipeline.py`](src/pipeline.py#L148), [`src/pipeline.py`](src/pipeline.py#L174)

`MAX_UPLOAD_BYTES` được kiểm tra khi route đọc `UploadFile`; lúc đó Starlette đã nhận và phân tích multipart request. Với định dạng không đọc được bằng SoundFile, nhánh PyAV giải mã toàn bộ audio thành danh sách frame rồi ghép vào RAM trước khi kiểm tra thời lượng. Vì thế giới hạn 20 MiB và `max_audio_seconds` không ngăn được chi phí xử lý/giải mã trước bước xác thực thời lượng. Nhiều request đồng thời còn có thể khuếch đại mức dùng CPU/RAM.

**Đề xuất:** đặt giới hạn body ở reverse proxy/ASGI server hoặc middleware trước parser multipart; xác thực metadata/container trước khi giải mã; giải mã theo luồng và dừng ngay khi vượt duration/sample cap; giới hạn số inference đồng thời và thời gian xử lý.

### P1 — Safeguard có thể xác nhận `CALM` bằng các giá trị đo thất bại được đổi thành 0

**Vị trí:** [`src/acoustic_extractor.py`](src/acoustic_extractor.py#L77), [`src/acoustic_extractor.py`](src/acoustic_extractor.py#L93), [`src/acoustic_extractor.py`](src/acoustic_extractor.py#L109), [`src/pipeline.py`](src/pipeline.py#L447)

Khi Praat không tính được jitter, shimmer hoặc HNR, extractor ghi `0.0`, giống một phép đo hợp lệ. Safeguard sau đó xem jitter/shimmer thấp là dấu hiệu bình tĩnh. Điều kiện kết luận chỉ cần F0 và jitter đạt ngưỡng cộng với ít nhất ba trên bốn cờ; vì vậy các giá trị lỗi bị đổi thành 0 có thể giúp hạ điểm mô hình xuống vùng CALM. Hệ quả là đầu vào nhiễu/khó đo có thể bị diễn giải thành “đã xác nhận bình tĩnh”. Ngưỡng F0 cố định cũng không có trạng thái chưa xác định cho một số giọng nằm ngoài các khoảng hard-code.

**Đề xuất:** giữ trạng thái thiếu/không hợp lệ riêng (NaN có kiểm soát hoặc `None` kèm quality flag); không dùng phép đo không hợp lệ trong safeguard, baseline hay XAI; trả `UNCERTAIN`/`INSUFFICIENT_AUDIO` khi chất lượng không đạt. Bỏ ngôn ngữ “triệt tiêu cảnh báo giả” nếu chưa có đánh giá độc lập chứng minh.

### P1 — Nhãn “stress” hiện là proxy từ cảm xúc diễn xuất, chưa xác nhận khả năng đo stress

**Vị trí:** [`data/prepare_manifest.py`](data/prepare_manifest.py#L23), [`data/prepare_manifest.py`](data/prepare_manifest.py#L37), [`README.md`](README.md#L3)

Manifest ánh xạ neutral/calm sang `normal` và angry/fearful (có thể thêm sad/disgust) sang `stress`. Mô hình vì vậy được huấn luyện phân biệt các nhóm cảm xúc trong RAVDESS; actor-disjoint split giúp kiểm tra người nói chưa thấy nhưng không biến nhãn cảm xúc thành nhãn stress sinh lý/tâm lý. Các chỉ số accuracy/recall hiện nêu trong README chỉ mô tả bài toán proxy này, không phải độ chính xác phát hiện stress ngoài đời. Safeguard còn có thể ghi kết quả CALM như một khẳng định chắc chắn hơn dữ liệu hỗ trợ.

**Đề xuất:** đặt tên kết quả là ước lượng cảm xúc/arousal proxy trong UI và tài liệu; trước khi tuyên bố đo stress, cần tập dữ liệu có nhãn stress phù hợp, quy trình gán nhãn được mô tả, tập đánh giá ngoài miền và báo cáo hiệu năng theo giới tính, chất lượng thu âm, ngôn ngữ và thiết bị. Công bố bất định và phạm vi sử dụng.

### P2 — API nhận audio tối đa 300 giây nhưng mô hình chỉ dùng tối đa 180 giây

**Vị trí:** [`configs/hparams_inference.yaml`](configs/hparams_inference.yaml#L4), [`src/stress_model.py`](src/stress_model.py#L49)

Pipeline chấp nhận audio đến 300 giây, nhưng `StressModel.predict` chỉ tạo chunk đến 180 giây đầu. Các phần cuối ngắn hơn 3 giây bị loại; chunk ngắn hơn 30 giây vẫn được lấy trung bình với trọng số ngang chunk đủ 30 giây. Kết quả không đại diện cho toàn bộ file và phụ thuộc vào vị trí nội dung trong audio.

**Đề xuất:** hoặc giảm giới hạn API xuống giới hạn thực được hỗ trợ, hoặc xử lý mọi chunk trong giới hạn duration và gộp theo số mẫu/thời lượng; trả thông tin phần audio đã dùng nếu có giới hạn cứng.

### P2 — Ngưỡng xác minh người nói không thống nhất giữa cấu hình, tài liệu và kiểm tra

**Vị trí:** [`configs/hparams_inference.yaml`](configs/hparams_inference.yaml#L23), [`src/pipeline.py`](src/pipeline.py#L45), [`README.md`](README.md#L108), [`tests/test_phase5.py`](tests/test_phase5.py#L167)

Cấu hình inference dùng ngưỡng `0.50` cho toàn bộ xác minh; README mô tả cả `0.70` và `0.50` theo ngữ cảnh, còn bài kiểm tra đòi similarity người cùng speaker đạt ít nhất `0.70` và người khác speaker không được xác minh. Ngưỡng thấp hơn có thể nhận nhầm người có similarity từ 0.50 đến 0.70, trong khi kết quả kiểm tra hiện không xác nhận cấu hình production đang dùng ngưỡng nào.

**Đề xuất:** dùng một cấu hình có tên/ngữ cảnh rõ ràng, hiệu chuẩn ngưỡng trên tập verification riêng với FAR/FRR mục tiêu, rồi dùng cùng giá trị đó trong API, tài liệu và kiểm tra.

### P2 — Tính năng tự nhận diện người nói được mô tả nhưng chưa nối vào luồng API

**Vị trí:** [`README.md`](README.md#L107), [`src/speaker_calibrator.py`](src/speaker_calibrator.py#L93), [`src/pipeline.py`](src/pipeline.py#L337)

`find_best_match` hiện là helper có kiểm tra trong script phase 4, nhưng `analyze` chỉ nạp baseline theo `user_id` được gửi lên; nếu thiếu ID thì xử lý là anonymous và không tìm trong danh sách profile. Vì vậy mô tả tự nhận diện trong README chưa phản ánh hành vi dashboard/API.

**Đề xuất:** hoặc nối một luồng nhận diện có ngưỡng/consent rõ và test chống nhận nhầm, hoặc sửa tài liệu/UI để nói người dùng phải nhập ID trước khi baseline cá nhân được áp dụng.

### P2 — Health check báo “healthy” dù các model cần thiết chưa sẵn sàng; kiểm tra phase 5 phụ thuộc tài sản ngoài Git

**Vị trí:** [`api/main.py`](api/main.py#L89), [`tests/test_phase5.py`](tests/test_phase5.py#L30), [`tests/test_phase5.py`](tests/test_phase5.py#L50), [`.gitignore`](.gitignore#L8)

`/health` báo `healthy` khi biến pipeline khác `None`, kể cả khi emotion model không tải được hoặc stress checkpoint chưa có. Trong khi đó `test_phase5.py` bắt buộc cả hai model sẵn sàng và đọc audio RAVDESS; dataset, checkpoint, kết quả huấn luyện và model tải về đều không nằm trong repo. Một clone mới không thể chạy bài kiểm tra end-to-end nếu chưa tải dữ liệu và huấn luyện/tải đủ artifact. Bài kiểm tra còn tạo `test_phase5_pilot` mà không dọn profile sau khi chạy.

**Đề xuất:** tách liveness khỏi readiness và trả từng trạng thái với HTTP/status rõ; tạo kiểm tra API nhẹ dùng model giả và fixture âm thanh nhỏ; giữ E2E nặng thành nhóm riêng có điều kiện dữ liệu/model; dọn baseline test trong `finally`.

### P2 — MetricGAN+ được tải khi khởi động nhưng không chạy trong pipeline inference

**Vị trí:** [`src/pipeline.py`](src/pipeline.py#L61), [`src/enhancer.py`](src/enhancer.py#L21), [`src/enhancer.py`](src/enhancer.py#L40)

Pipeline khởi tạo `AudioEnhancer`, dẫn đến tải model MetricGAN+, nhưng luồng `analyze` làm sạch/resample bằng `_validate_and_prepare_audio` và không gọi `clean_audio`. Điều này làm tăng thời gian khởi động và bộ nhớ, đồng thời README mô tả enhancer như một phần xử lý thực tế nhưng đường inference hiện không dùng denoiser.

**Đề xuất:** quyết định rõ enhancer có thuộc pipeline production không. Nếu có, áp dụng và đánh giá tác động lên tất cả model; nếu chưa, bỏ tải bắt buộc hoặc chuyển sang tùy chọn/lazy loading và sửa tài liệu.

### P2 — Dashboard mặc định hiển thị CALM cho trạng thái lạ và chèn explanation bằng HTML

**Vị trí:** [`frontend/app.js`](frontend/app.js#L225), [`frontend/app.js`](frontend/app.js#L303)

`renderResult` chỉ xử lý rõ `HIGH_STRESS`, `MODERATE_STRESS` và `UNCALIBRATED`; trạng thái bị thiếu hoặc không nhận diện rơi vào nhánh CALM. Gauge cũng dùng màu xanh cho mọi status khác. Đồng thời explanation từ response được đưa trực tiếp vào `innerHTML`. Response hiện tại phần lớn do backend tạo, nhưng frontend cho phép cấu hình API URL; nếu response không tin cậy, đây là DOM-XSS và có thể chạy script trong origin của dashboard.

**Đề xuất:** dùng trạng thái mặc định `UNKNOWN`/trung tính, không suy score null thành 0 hoặc CALM; dựng nội dung explanation bằng DOM và `textContent`, không ghép HTML từ response.

### P2 — Giới hạn định dạng/thời lượng và kết quả benchmark trong tài liệu khó tái lập

**Vị trí:** [`README.md`](README.md#L66), [`README.md`](README.md#L91), [`frontend/index.html`](frontend/index.html#L42), [`configs/hparams_inference.yaml`](configs/hparams_inference.yaml#L4), [`requirements.txt`](requirements.txt#L1)

README phần hướng dẫn API ghi WAV/FLAC/OGG và tối đa 30 giây; phần dashboard/config lại cho phép nhiều định dạng và 300 giây. README công bố các metric test nhưng artifact metric/checkpoint nằm trong `results/` bị ignore nên không có sẵn trong repo để xác nhận. Requirements chỉ đặt cận dưới, không khóa các phiên bản đã chạy cùng SpeechBrain/Torch/Torchaudio.

**Đề xuất:** thống nhất tài liệu theo cấu hình thực; kèm script tạo manifest và cấu hình, phiên bản dataset/dependency, checkpoint hash cùng file metric tái tạo được; bổ sung lockfile hoặc bộ phiên bản đã xác minh.

### P3 — Một recording duy nhất tạo baseline với độ lệch chuẩn suy ra bằng công thức cố định

**Vị trí:** [`src/pipeline.py`](src/pipeline.py#L222), [`src/pipeline.py`](src/pipeline.py#L232)

Baseline được lập từ một đoạn thu và các độ lệch chuẩn của jitter/shimmer/HNR được suy ra theo tỷ lệ cố định với giá trị đo, cộng mức sàn. Đây không phải độ biến thiên quan sát được của người nói; một recording ngắn hoặc nhiễu có thể tạo z-score/XAI quá nhạy hoặc quá yếu.

**Đề xuất:** thu nhiều đoạn bình tĩnh trong các phiên khác nhau, kiểm tra độ dài/chất lượng, ước lượng phân bố thực và cho phép báo “chưa đủ dữ liệu baseline”.

## Đánh giá theo khu vực

| Khu vực | Đánh giá |
|---|---|
| Cấu trúc mã | Module theo trách nhiệm tương đối rõ; pipeline đang gánh nhiều quyết định xử lý, hiệu chuẩn và diễn giải kết quả nên nên tách scoring/quality gate khỏi orchestration. |
| Dữ liệu và mô hình | Có chia actor độc lập và chuẩn hóa feature theo thống kê train; construct validity của nhãn stress là khoảng trống lớn nhất. |
| API | Có dọn file tạm và trả lỗi có kiểm soát; thiếu auth, giới hạn sớm request và readiness chính xác. |
| Giao diện | Có luồng upload, demo, JSON và xử lý DOM; hiển thị stress hiện mang vẻ kết luận chắc chắn hơn nền tảng đánh giá hỗ trợ. |
| Huấn luyện và phát hành | Có script tạo manifest, huấn luyện và xuất metric; thiếu artifact/reproducibility bundle và bộ dependency khóa phiên bản. |

## Kế hoạch xử lý đề xuất

1. **Trước khi chia sẻ hoặc mở mạng:** thêm xác thực/ủy quyền, origin allow-list, giới hạn request trước multipart parser, giới hạn decode/concurrency.
2. **Trước khi tin kết quả:** không biến phép đo lỗi thành 0; bỏ ép `CALM`; thể hiện thiếu dữ liệu/bất định; điều chỉnh nhãn và cách gọi theo đúng dữ liệu RAVDESS.
3. **Trước khi phát hành ổn định:** sửa xử lý chunk 300 giây, thống nhất ngưỡng speaker, phân biệt liveness/readiness, bổ sung API tests nhẹ có fixture và cleanup.
4. **Để tái lập:** đồng bộ README/UI/config; khóa dependency; lưu metric, manifest recipe, phiên bản dữ liệu và checkpoint/hash có thể đối chiếu.
5. **Để hoàn thiện baseline:** đăng ký từ nhiều recording có quality gate và bổ sung quản lý/xóa profile.

## Giới hạn của lượt review

Đây là review tĩnh tại commit `d9c302b`. Không chạy tests, không cài dependency, không tải dataset/model và không đánh giá thực nghiệm chất lượng mô hình. Các nhận xét về hiệu năng/độ chính xác là rủi ro suy ra từ luồng mã, không thay thế đo kiểm trên dữ liệu thực.
