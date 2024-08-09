from app.model_loader import load_model
from app.model_inference import model_inference

if __name__ == '__main__':
    model = load_model()
    input_file_path = "/home/supers/SAG/TTTTT/prod_code/uploads/audio_6.wav"
    output_dir = "/home/supers/SAG/TTTTT/prod_code/output/"

    model_inference(model, input_file_path, output_dir)

