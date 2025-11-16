const express = require('express');
const session = require('express-session');
const { v4: uuidv4 } = require('uuid');
const multer = require('multer');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');

// Global error handlers to ensure stack traces appear in Vercel logs
process.on('uncaughtException', (err) => {
    console.error('Uncaught exception:', err && err.stack ? err.stack : err);
});
process.on('unhandledRejection', (reason) => {
    console.error('Unhandled rejection:', reason && reason.stack ? reason.stack : reason);
});

const app = express();
const port = 3000;

// Session middleware setup
app.use(session({
    secret: 'cat-dog-classifier-secret',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set to true if using HTTPS
}));

// Allow EJS to render .html files so views can be renamed to .html
app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static('public'));

// Configure multer for memory storage
const upload = multer({
    storage: multer.memoryStorage(),
    fileFilter: function (req, file, cb) {
        if (!file.originalname.match(/\.(jpg|jpeg|png)$/)) {
            return cb(new Error('Only image files are allowed!'), false);
        }
        cb(null, true);
    }
});

// Initialize SQLite database with user_id and image blob
const dbPath = 'predictions.db';

// Check if we need to migrate the database
let needsMigration = false;
if (fs.existsSync(dbPath)) {
    const tempDb = new sqlite3.Database(dbPath);
    tempDb.get("SELECT sql FROM sqlite_master WHERE type='table' AND name='predictions'", [], (err, row) => {
        if (row && (!row.sql.includes('user_id') || !row.sql.includes('image_data'))) {
            console.log('Old database schema detected, recreating database...');
            tempDb.close();
            fs.unlinkSync(dbPath);
            needsMigration = true;
        }
    });
    tempDb.close();
}

const db = new sqlite3.Database(dbPath);
db.serialize(() => {
    db.run(`CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        prediction TEXT,
        confidence REAL,
        image_data BLOB,
        mime_type TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )`);
});

// Middleware to ensure user has an ID
app.use((req, res, next) => {
    if (!req.session.userId) {
        req.session.userId = uuidv4();
    }
    next();
});

// Routes
app.get('/', (req, res) => {
    db.all(
        'SELECT id, prediction, confidence, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10',
        [req.session.userId],
        (err, rows) => {
            res.render('index', { 
                recentPredictions: rows || [],
                userId: req.session.userId
            });
        }
    );
});

// Route to serve images from database
app.get('/image/:id', (req, res) => {
    // Verify user has access to this image
    db.get(
        'SELECT image_data, mime_type FROM predictions WHERE id = ? AND user_id = ?',
        [req.params.id, req.session.userId],
        (err, row) => {
            if (err || !row) {
                return res.status(404).send('Image not found');
            }
            
            res.writeHead(200, {
                'Content-Type': row.mime_type,
                'Content-Length': row.image_data.length
            });
            res.end(row.image_data);
        }
    );
});

app.post('/upload', upload.single('image'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        // Quick environment note for logs
        if (process.env.VERCEL) {
            console.log('Running in Vercel environment - note: Python & local storage may be unavailable');
        }

        // Check for Python availability (serverless environments often don't provide it)
        const pythonCheck = spawnSync('python', ['--version'], { encoding: 'utf8' });
        if (pythonCheck.error || pythonCheck.status !== 0) {
            console.error('Python runtime not available or returned error:', pythonCheck.error || pythonCheck.stderr || pythonCheck.stdout);
            return res.status(501).json({ error: 'Python runtime not available in this environment. Deploy backend to a full server that supports Python.' });
        }

        // Ensure model file exists
        const modelPath = path.join(__dirname, '..', 'cat_dog_classifier.pkl');
        if (!fs.existsSync(modelPath)) {
            console.error('Model file not found:', modelPath);
            return res.status(500).json({ error: 'Model file not found' });
        }

        const pythonScriptPath = path.join(__dirname, '..', 'predict_image_rf.py');
        if (!fs.existsSync(pythonScriptPath)) {
            console.error('Python script not found:', pythonScriptPath);
            return res.status(500).json({ error: 'Prediction script not found' });
        }

        // Create a temporary file from the uploaded buffer
        const tempFilePath = path.join(os.tmpdir(), `temp-${Date.now()}.jpg`);
        fs.writeFileSync(tempFilePath, req.file.buffer);

        const pythonProcess = spawn('python', [pythonScriptPath, tempFilePath]);

        let predictionData = '';

        pythonProcess.stdout.on('data', (data) => {
            predictionData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error('Python Error (stderr):', data.toString());
        });

        pythonProcess.on('close', (code) => {
            // Clean up the temporary file
            try { fs.unlinkSync(tempFilePath); } catch (e) { /* ignore */ }

            if (code !== 0) {
                console.error('Python process exited with code', code);
                return res.status(500).json({ error: 'Prediction failed' });
            }

            // Parse prediction result
            const match = predictionData.match(/->\s*(CAT|DOG)\s*\(confidence:\s*([\d.]+)/);
            if (match) {
                const [_, prediction, confidence] = match;

                // Save to database with user_id and image data
                db.run(
                    'INSERT INTO predictions (user_id, prediction, confidence, image_data, mime_type) VALUES (?, ?, ?, ?, ?)',
                    [req.session.userId, prediction, parseFloat(confidence), req.file.buffer, req.file.mimetype],
                    (err) => {
                        if (err) {
                            console.error('Database error:', err);
                        }
                    }
                );

                return res.json({ success: true, prediction: prediction, confidence: parseFloat(confidence) });
            } else {
                console.error('Could not parse prediction output:', predictionData);
                return res.status(500).json({ error: 'Could not parse prediction result' });
            }
        });
    } catch (err) {
        console.error('Upload handler error:', err && err.stack ? err.stack : err);
        return res.status(500).json({ error: 'Internal server error', detail: err.message });
    }
});

// Handle errors
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});