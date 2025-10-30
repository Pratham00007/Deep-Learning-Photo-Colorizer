

# 🖼️ Image Colorization using OpenCV DNN

## 🎯 Project Aim

Humara goal yeh hai ki **black and white (grayscale) image ko color image** me convert karna — using a **pre-trained Deep Learning model** from OpenCV.
Yeh model automatically guess karta hai ki kis pixel ko kaunsa color milna chahiye.

---

## 🧠 Concept in Short

* **Grayscale image** → sirf brightness (L channel)
* **Color image (LAB)** → brightness (L) + color info (a,b)
* Model predict karta hai `(a,b)` values from given L, aur hum use merge karke color image banaate hain.

---

## ⚙️ Steps in Code (Line by Line Explanation)

### 📂 Load Model & Files

```python
prototxt_path = r'models\colorization_deploy_v2.prototxt'
model_path = r'models\colorization_release_v2.caffemodel'
kernel_path = r'models\pts_in_hull.npy'
```

➡️ Yeh teen files model ke liye important hain:

* `.prototxt`: Model structure
* `.caffemodel`: Trained weights
* `.npy`: Color cluster centers

---

```python
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
```

➡️ Model ko memory me load karta hai — OpenCV ke DNN module ke through.

---

```python
points = np.load(kernel_path)
points = points.transpose().reshape(2,313,1,1)
```

➡️ 313 color clusters ko reshape karte hain taaki model me feed kiya ja sake.

---

```python
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313],2.606,dtype="float32")]
```

➡️ Yeh lines model ke specific layers ko initialize karte hain:

* `class8_ab`: cluster centers load karta hai
* `conv8_313_rh`: ek balancing constant set karta hai (2.606)

---

### 🖼️ Load & Convert Image

```python
bw_image = cv2.imread(image_path)
normalized = bw_image.astype('float32')/255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
```

➡️ Image load hoti hai, normalize hoti hai (0–1 range me), aur LAB color space me convert hoti hai.

---

### ✂️ Extract L Channel

```python
resized = cv2.resize(lab,(224,224))
L = cv2.split(resized)[0]
L -= 50
```

➡️ Image ko resize karte hain (224×224 — model input size)
➡️ LAB me se sirf Lightness (L) channel nikalte hain
➡️ Mean subtraction (−50) se data normalize hota hai.

---

### 🤖 Model Inference

```python
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0,:,:,:].transpose((1,2,0))
```

➡️ `blobFromImage` model ke input format me data convert karta hai.
➡️ `net.forward()` model ko run karta hai aur `(a,b)` color channels predict karta hai.

---

### 🔁 Resize & Merge

```python
ab = cv2.resize(ab,(bw_image.shape[1],bw_image.shape[0]))
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis],ab),axis=2)
```

➡️ Predicted `(a,b)` ko resize karke original image size me fit karte hain.
➡️ `L` + `(a,b)` ko merge karke complete LAB image banaate hain.

---

### 🎨 Convert to BGR

```python
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)
colorized = (255.0 * colorized).astype("uint8")
```

➡️ LAB image ko back to BGR (normal color image) me convert karte hain.
➡️ Values ko 0–255 range me clip aur convert karte hain display ke liye.

---

### 🖥️ Show Result

```python
cv2.imshow("Black & White Image", bw_image)
cv2.imshow("Colorized Image", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

➡️ Do windows khulti hain: ek grayscale image ke liye, ek colorized output ke liye.
➡️ Program user ke key press ka wait karta hai aur phir windows close karta hai.

---

## 🧩 Summary (In Short)

| Step            | Description                                    |
| --------------- | ---------------------------------------------- |
| Load model      | Pretrained Deep Learning model load karna      |
| Prepare input   | Grayscale image se Lightness (L) extract karna |
| Run model       | Model se a,b color channels predict karna      |
| Merge & convert | L + ab combine karke color image banana        |
| Show result     | Black & white aur colorized image dikhana      |

---

## 💡 Key Concepts

* **LAB color space** → L (brightness) + A,B (color info)
* **DNN** → Deep Neural Network trained on ImageNet colors
* **OpenCV DNN module** → Lightweight inference engine (no TensorFlow/PyTorch needed)
