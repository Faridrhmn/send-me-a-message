from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Menggunakan os.urandom untuk mengenkripsi sesi (cookies login) secara acak dan sangat aman
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

import random

try:
    from langdetect import detect
except ImportError:
    detect = lambda text: 'id'

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    vader_analyzer = None

# Setup Local AI Model for Offline & Fast Generation
try:
    from transformers import pipeline
    print("Memuat model AI lokal (Offline)...")
    # Menggunakan model Sentimen Analisis Bahasa Indonesia yang sangat kecil dan ringan
    sentiment_analyzer = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
except Exception as e:
    print(f"Model lokal belum terinstal atau gagal dimuat: {e}")
    sentiment_analyzer = None

# Free AI / Contextual Reply Function using Local Model (Bilingual)
def generate_reply(message_text):
    # --- INDONESIAN REPLIES (Merespons pujian/kritik untuk Pemilik Web) ---
    id_positive = [
        "Terima kasih banyak atas kata-kata baikmu! Ini benar-benar membuat hariku lebih cerah. ✨",
        "Wah, aku sangat menghargai pesan positif ini. Terima kasih banyak ya! 😊",
        "Pesanmu sangat berarti buatku. Terima kasih sudah mampir dan memberikan dukungan! ❤️",
        "Aww, kamu baik sekali! Terima kasih atas energi positifnya! 🚀"
    ]
    id_negative = [
        "Terima kasih atas tanggapan dan kejujuranmu. Aku sangat menghargainya dan akan berusaha lebih baik lagi. 🙏",
        "Maaf jika ada yang kurang berkenan. Masukan darimu sangat berarti untuk perkembanganku. 💪",
        "Terima kasih sudah meluangkan waktu untuk memberikan kritik. Ini akan jadi bahan evaluasi buatku! 🌱",
        "Aku membaca pesanmu. Terima kasih atas kejujurannya, aku akan terus belajar menjadi lebih baik. 🤝"
    ]
    id_neutral = [
        "Halo! Terima kasih banyak sudah mampir dan meninggalkan pesan di sini. 💌",
        "Senang sekali melihatmu mampir! Pesanmu sudah tersimpan dengan aman. 🌻",
        "Terima kasih sudah meninggalkan jejak di sini. Semoga harimu menyenangkan! 📝",
        "Pesan diterima! Terima kasih banyak atas waktunya. 🕊️"
    ]

    # --- ENGLISH REPLIES (Responding to compliments/critique for the Web Owner) ---
    en_positive = [
        "Thank you so much for the kind words! It truly makes my day. ✨",
        "Wow, I really appreciate this positive message. Thank you! 😊",
        "Your message means a lot to me. Thanks for stopping by and showing support! ❤️",
        "Aww, you are too kind! Thanks for the positive energy! 🚀"
    ]
    en_negative = [
        "Thank you for your honest feedback. I really appreciate it and will try to do better. 🙏",
        "I'm sorry if I fell short. Your input means a lot for my personal growth. 💪",
        "Thanks for taking the time to share your critique. I'll use this to improve myself! 🌱",
        "I've read your message. Thanks for your honesty, I will keep learning and improving. 🤝"
    ]
    en_neutral = [
        "Hello! Thank you so much for dropping by and leaving a message. 💌",
        "So glad to see you here! Your message has been safely saved. 🌻",
        "Thanks for leaving your mark here. Hope you have a wonderful day! 📝",
        "Message received! Thank you so much for your time. 🕊️"
    ]

    # Detect language
    try:
        # Provide enough context for langdetect
        lang = detect(message_text + " " + message_text)
    except:
        lang = 'id'
        
    is_english = (lang == 'en')
    
    pos_replies = en_positive if is_english else id_positive
    neg_replies = en_negative if is_english else id_negative
    neu_replies = en_neutral if is_english else id_neutral
    
    # Sentiment Analysis
    try:
        label = 'neutral'
        if is_english and vader_analyzer:
            score = vader_analyzer.polarity_scores(message_text)
            compound = score['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
        elif not is_english and sentiment_analyzer:
            result = sentiment_analyzer(message_text)[0]
            label = result['label']
            
        if label == 'positive':
            return random.choice(pos_replies)
        elif label == 'negative':
            return random.choice(neg_replies)
        else:
            return random.choice(neu_replies)
    except Exception as e:
        print(f"Local AI Error: {e}")
        
    # Fallback to simple rule-based if AI models fail
    text = message_text.lower()
    if is_english:
        if any(word in text for word in ['sad', 'disappointed', 'cry', 'bad', 'fail', 'broken', 'tired', 'hurt']):
            return random.choice(neg_replies)
        elif any(word in text for word in ['happy', 'glad', 'success', 'good', 'cool', 'love', 'fun']):
            return random.choice(pos_replies)
    else:
        if any(word in text for word in ['sedih', 'kecewa', 'menangis', 'buruk', 'gagal', 'hancur', 'capek', 'lelah', 'sakit']):
            return random.choice(neg_replies)
        elif any(word in text for word in ['senang', 'bahagia', 'sukses', 'berhasil', 'baik', 'keren', 'cinta', 'seru']):
            return random.choice(pos_replies)
            
    return random.choice(neu_replies)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/submit', methods=['POST'])
def submit_message():
    data = request.get_json()
    message_content = data.get('message', '').strip()
    
    if not message_content:
        return jsonify({'error': 'Message cannot be empty'}), 400
        
    new_message = Message(content=message_content)
    db.session.add(new_message)
    db.session.commit()
    
    reply = generate_reply(message_content)
    return jsonify({'reply': reply})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('admin'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('admin'))
        else:
            flash('Incorrect username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/admin')
@login_required
def admin():
    messages = Message.query.order_by(Message.timestamp.desc()).all()
    return render_template('admin.html', messages=messages)

@app.route('/admin/delete/<int:msg_id>', methods=['POST'])
@login_required
def delete_message(msg_id):
    msg = Message.query.get_or_404(msg_id)
    db.session.delete(msg)
    db.session.commit()
    flash('Message successfully deleted.', 'success')
    return redirect(url_for('admin'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Backup Logic
def backup_messages():
    with app.app_context():
        messages = Message.query.all()
        backup_path = os.path.join(app.root_path, 'backups.txt')
        with open(backup_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Backup on {datetime.datetime.now()} ---\n")
            for msg in messages:
                f.write(f"[{msg.timestamp}] {msg.content}\n")
        print(f"Backup completed at {datetime.datetime.now()}")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create or update default admin user from environment variables
        admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
        admin_password = os.environ.get('ADMIN_PASSWORD', 'defaultpassword')
        
        admin_user = User.query.first()
        if not admin_user:
            hashed_pw = generate_password_hash(admin_password, method='pbkdf2:sha256')
            admin_user = User(username=admin_username, password=hashed_pw)
            db.session.add(admin_user)
            db.session.commit()
        else:
            admin_user.username = admin_username
            admin_user.password = generate_password_hash(admin_password, method='pbkdf2:sha256')
            db.session.commit()            
    # Schedule backup every 1 month (approx 30 days)
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=backup_messages, trigger="interval", days=30)
    scheduler.start()
    
    try:
        app.run(debug=True, port=5000)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
