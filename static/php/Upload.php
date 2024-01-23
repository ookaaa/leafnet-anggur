<?php
// Periksa apakah file gambar telah dikirim
if(isset($_FILES['image'])) {
    $uploadDir = 'uploads/';
    $fileName = basename($_FILES['image']['name']);
    $targetPath = $uploadDir . $fileName;
    $fileType = pathinfo($targetPath, PATHINFO_EXTENSION);

    // Periksa apakah file adalah gambar
    $allowedTypes = array('jpg', 'jpeg', 'png');
    if (in_array($fileType, $allowedTypes)) {
        // Pindahkan file ke direktori upload
        if (move_uploaded_file($_FILES['image']['tmp_name'], $targetPath)) {
            // Panggil model deteksi daun buah apel (disini kamu perlu mengganti dengan kode untuk menjalankan model deteksi daun)
            // $result = runLeafDetectionModel($targetPath);

            // Contoh hasil deteksi (disini kamu perlu mengganti dengan hasil deteksi dari model)
            $result = array(
                'message' => 'Daun pada buah apel terdeteksi.',
                'image_path' => $targetPath
            );

            echo json_encode($result);
        } else {
            $result = array(
                'message' => 'Terjadi kesalahan saat mengunggah gambar.'
            );
            echo json_encode($result);
        }
    } else {
        $result = array(
            'message' => 'File yang diunggah bukan gambar.'
        );
        echo json_encode($result);
    }
} else {
    $result = array(
        'message' => 'Tidak ada gambar yang dikirim.'
    );
    echo json_encode($result);
}
?>

<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
  $color = $_POST["color"];
  // Simpan nilai warna ke dalam file konfigurasi
  file_put_contents("config.txt", $color);
}
?>

<!DOCTYPE html>
<html>
<head>
  <title>Ubah Warna</title>
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
  <style>
    body {
      background-color: <?php echo file_get_contents("config.txt", true); ?>;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Ubah Warna</h2>
    <form method="POST" action="<?php echo $_SERVER['PHP_SELF']; ?>">
      <div class="form-group">
        <label for="color">Pilih Warna</label>
        <input type="color" id="color" name="color" value="#ff3b30" class="form-control">
      </div>
      <div class="form-group">
        <button type="submit" class="btn btn-primary">Simpan</button>
      </div>
    </form>
  </div>
</body>
</html>

