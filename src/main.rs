use opencv::{
    core::{Vector, Size, Scalar, Point},
    highgui, imgproc, objdetect, prelude::*, videoio,
};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use uuid::Uuid;

const DATABASE: &str = "./face_data.json";

#[derive(Serialize, Deserialize, Clone)]
struct FaceEntry {
    id: String,
    features: Vec<f32>,
    allowed: bool, // true: Zugang erlaubt, false: Zugang verweigert
}

impl FaceEntry {
    fn new(features: Vec<f32>, allowed: bool) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            features,
            allowed,
        }
    }
}

/// Lädt bekannte Gesichter aus der JSON-Datei
fn load_face_data() -> Vec<FaceEntry> {
    let mut file = OpenOptions::new()
        .read(true)
        //.create(true)
        .open(DATABASE)
        .expect("Konnte face_data.json nicht öffnen!");
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    serde_json::from_str(&content).unwrap_or_else(|_| vec![])
}

/// Speichert neue Gesichtsdaten in die JSON-Datei
fn save_face_data(entry: &FaceEntry) {
    let mut data = load_face_data();
    data.push(entry.clone());
    let json_data = serde_json::to_string_pretty(&data).expect("Fehler beim Serialisieren");
    let mut file = File::create(DATABASE).expect("Fehler beim Erstellen von face_data.json");
    file.write_all(json_data.as_bytes())
        .expect("Fehler beim Schreiben in die Datei");
}

/// Berechnet die Kosinus-Ähnlichkeit zwischen zwei Feature-Vektoren
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot: f32 = v1.iter().zip(v2).map(|(a, b)| a * b).sum();
    let mag1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let mag2: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();
    dot / (mag1 * mag2)
}

/// Sucht in der Datenbank nach einem bekannten Gesicht (auf Basis des Feature-Vektors)
fn find_existing_face(features: &[f32]) -> Option<FaceEntry> {
    let known_faces = load_face_data();
    known_faces
        .iter()
        .find(|face| cosine_similarity(&face.features, features) > 0.9)
        .cloned()
}

/// Gesichtserkennung mithilfe der Kamera und OpenCV
fn recognize_face_from_camera() {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .expect("Kamera konnte nicht geöffnet werden");
    let mut face_cascade =
        objdetect::CascadeClassifier::new("./haarcascade_frontalface_default.xml")
            .expect("Fehler beim Laden des Haarcascades");

    if !cam.is_opened().unwrap() {
        panic!("Kamera nicht gefunden");
    }

    let mut frame = Mat::default();
    loop {
        cam.read(&mut frame).unwrap();
        let mut gray = Mat::default();
        // Wir verwenden hier unsafe { std::mem::zeroed() } als Workaround für den AlgorithmHint-Parameter.
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            unsafe { std::mem::zeroed() },
        )
            .unwrap();

        let mut faces = Vector::<opencv::core::Rect>::new();
        face_cascade
            .detect_multi_scale(
                &gray,
                &mut faces,
                1.1,
                3,
                objdetect::CASCADE_SCALE_IMAGE,
                Size::new(30, 30),
                Size::new(200, 200),
            )
            .unwrap();

        for face in faces.iter() {
            // Extrahiere den Bereich des Gesichts und klone ihn
            let roi_box = Mat::roi(&gray, face).unwrap();
            let face_region = roi_box.try_clone().unwrap();
            let features = extract_features(&face_region);

            let mut access_allowed = true;

            let mut draw_color = Scalar::new(0.0, 255.0, 0.0, 0.0); // grün: Zugang erlaubt

            // Prüfe, ob das Gesicht bereits in der Datenbank vorhanden ist
            if let Some(existing_face) = find_existing_face(&features) {
                access_allowed = existing_face.allowed;
                if access_allowed {
                    println!("Willkommen zurück!");
                    draw_color = Scalar::new(0.0, 255.0, 0.0, 0.0); // grün
                } else {
                    println!("ALERT: Zugang verweigert! Unbefugtes Betreten!");
                    draw_color = Scalar::new(0.0, 0.0, 255.0, 0.0); // rot
                }
            } else {
                // Erstmalige Erkennung: Prompt zur Zugangskontrolle
                println!("Neue Person erkannt. Zugang gewähren? (j/n): ");
                let mut response = String::new();
                io::stdin()
                    .read_line(&mut response)
                    .expect("Fehler beim Lesen der Eingabe");
                access_allowed = response.trim().to_lowercase() == "j";
                if access_allowed {
                    println!("Zugang erlaubt. Willkommen!");
                    draw_color = Scalar::new(0.0, 255.0, 0.0, 0.0); // grün
                } else {
                    println!("ALERT: Zugang verweigert! Unbefugtes Betreten!");
                    draw_color = Scalar::new(0.0, 0.0, 255.0, 0.0); // rot
                }
                let new_entry = FaceEntry::new(features, access_allowed);
                save_face_data(&new_entry);
            }
            // Zeichne den Rahmen um das erkannte Gesicht
            imgproc::rectangle(&mut frame, face, draw_color, 2, imgproc::LINE_8, 0)
                .unwrap();

            // Falls der Zugang verweigert ist, füge oberhalb des Rahmens den Text hinzu
            if !access_allowed {
                let text = "Zugang verweigert";
                // Positioniere den Text etwas oberhalb des Rechtecks
                let org = Point::new(face.x, if face.y - 10 > 0 { face.y - 10 } else { face.y });
                imgproc::put_text(
                    &mut frame,
                    text,
                    org,
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    false,
                )
                    .unwrap();
            }
        }

        highgui::imshow("Gesichtserkennung", &frame).unwrap();
        if highgui::wait_key(10).unwrap() == 27 {
            // ESC-Taste zum Beenden
            break;
        }
    }
}

/// Extrahiere Merkmale aus einem Gesicht (Dummy-Implementierung)
fn extract_features(face: &Mat) -> Vec<f32> {
    let mut resized = Mat::default();
    imgproc::resize(
        face,
        &mut resized,
        Size::new(100, 100),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
        .unwrap();

    // Erstelle einen Dummy-Feature-Vektor (normiere Pixelwerte)
    resized
        .data_bytes()
        .unwrap()
        .iter()
        .map(|&x| x as f32 / 255.0)
        .collect()
}

fn main() {
    recognize_face_from_camera();
}
