# Projek Deteksi Anomali (ADBench Angle III) - Kelompok 1

Struktur: data/ (file .npz berisi X dan y), src/ (generator, models, pipeline), tests/, results/.

Cara pakai:
1. pip install -r requirements.txt
2. Taruh 12 file .npz ADBench di data/
3. cd src && python dataset_table.py ../data
4. cd .. && python -m pytest tests -q
5. cd src && python run_baseline.py ../data ../results/baseline.csv
6. cd src && python pipeline.py ../data ../results/hasil_lengkap.csv   (eksperimen penuh)
