# Crisis MAS - Web Tools for LLM Training

User-friendly web interface for creating and managing emergency response scenarios and expert profiles without needing to manually edit JSON files.

## Features

### No JSON Editing Required
- Clean, intuitive web forms replace complex JSON syntax
- Real-time validation ensures data quality
- Auto-generated IDs and timestamps

### Scenario Management
- Create emergency response scenarios with questions and answers
- Support for Greek (🇬🇷) and English (🇬🇧) languages
- Categorize by type: Firefighting, Police, Medical, Search & Rescue, Disaster Response
- Severity levels: Low, Medium, High, Critical
- Tag scenarios for easy organization
- Link scenarios to expert profiles

### Expert Profile Management
- Comprehensive expert profiles
- Track experience, specializations, certifications
- Multi-language support
- Availability status tracking
- Contact information management

### Data Export
- Export all data as JSON for LLM training pipeline
- API endpoints for programmatic access
- Compatible with existing training tools

## Quick Start

### Installation

1. **Navigate to web_tools directory**
   ```bash
   cd web_tools
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web server**
   ```bash
   python app.py
   ```

4. **Open your browser**
   ```
   http://localhost:5000
   ```

That's it! The application will create sample data automatically on first run.

## Usage

### Creating a Scenario

1. Click **"Scenarios"** in the navigation menu
2. Click **"Create New Scenario"**
3. Fill in the form:
   - **Question**: The emergency scenario question (can be in Greek or English)
   - **Expert Answer**: Detailed response from an expert
   - **Category**: Select type (Firefighting, Police, Medical, etc.)
   - **Severity**: Select severity level
   - **Location**: Optional location information
   - **Language**: Select Greek or English
   - **Resources**: Equipment or personnel needed
   - **Expert**: Link to an expert profile (optional)
   - **Tags**: Keywords for organization (comma-separated)
4. Click **"Create Scenario"**

### Creating an Expert Profile

1. Click **"Experts"** in the navigation menu
2. Click **"Create New Expert"**
3. Fill in the form:
   - **Basic Info**: Name, Role, Organization, Location
   - **Experience**: Years of experience
   - **Skills**: Specializations, Certifications, Languages (comma-separated)
   - **Contact**: Email and phone (optional)
   - **Availability**: Current availability status
   - **Bio**: Brief biography or notes
4. Click **"Create Expert"**

### Editing and Deleting

- Click the **"Edit"** button on any scenario or expert card
- Click **"Delete"** to remove (with confirmation)
- Changes are saved immediately to JSON files

### Exporting Data

**Option 1: Web Interface**
- Click **"Export"** in the navigation menu
- Downloads complete JSON file

**Option 2: API**
```bash
# Get all scenarios
curl http://localhost:5000/api/scenarios

# Get all experts
curl http://localhost:5000/api/experts

# Export everything
curl http://localhost:5000/api/export > crisis_mas_data.json
```

## File Structure

```
web_tools/
├── app.py                  # Flask application (main server)
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── templates/             # HTML templates
│   ├── index.html        # Home page
│   ├── scenarios.html    # Scenarios list
│   ├── scenario_form.html # Scenario create/edit form
│   ├── experts.html      # Experts list
│   └── expert_form.html  # Expert create/edit form
│
├── static/               # Static assets
│   ├── css/
│   │   └── style.css    # Custom styles
│   └── js/
│       └── main.js      # Client-side JavaScript
│
└── data/                # JSON data storage
    ├── scenarios.json   # Scenarios database
    └── experts.json     # Expert profiles database
```

## Data Format

### Scenario JSON Structure

```json
{
  "id": "scenario_001",
  "question": "Emergency question?",
  "answer": "Expert response...",
  "category": "firefighting",
  "location": "Athens",
  "severity": "high",
  "resources_required": "Fire trucks, etc.",
  "expert_id": "expert_001",
  "language": "el",
  "tags": ["fire", "emergency"],
  "created_at": "2024-11-14T10:00:00",
  "updated_at": "2024-11-14T10:00:00"
}
```

### Expert Profile JSON Structure

```json
{
  "id": "expert_001",
  "name": "John Doe",
  "role": "Senior Firefighter",
  "organization": "Athens Fire Department",
  "location": "Athens, Greece",
  "experience_years": 15,
  "specializations": ["Wildfire", "Urban Fire"],
  "certifications": ["Advanced Firefighting"],
  "email": "john@example.com",
  "phone": "+30 123 456 7890",
  "languages": ["Greek", "English"],
  "availability": "available",
  "bio": "Brief biography...",
  "created_at": "2024-11-14T09:00:00",
  "updated_at": "2024-11-14T09:00:00"
}
```

## Integration with LLM Training Pipeline

The web tools generate JSON files compatible with the LLM training pipeline:

1. **Create scenarios and experts** using the web interface
2. **Export data** via the Export button or API
3. **Use exported JSON** with training tools:
   ```bash
   # Copy to training pipeline
   cp data/scenarios.json "../LLM Training/data/training_scenarios.json"
   cp data/experts.json "../LLM Training/data/expert_profiles.json"
   
   # Or use the training tools directly
   llm-collect --input data/scenarios.json
   ```

## Production Deployment

### Using Gunicorn (Recommended)

```bash
# Install gunicorn
pip install gunicorn

# Run with 4 worker processes
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

Create a `.env` file for configuration:

```env
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DATA_DIR=/path/to/data
```

### Using Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Security Considerations

1. **Change the secret key** in `app.py` before production deployment
2. **Use HTTPS** in production (configure with a reverse proxy like Nginx)
3. **Implement authentication** if deploying on public networks
4. **Backup data files** regularly (`data/scenarios.json`, `data/experts.json`)
5. **Set file permissions** appropriately on the data directory

## Troubleshooting

### Port Already in Use
```bash
# Change port in app.py or use environment variable
PORT=8080 python app.py
```

### Permission Denied on Data Files
```bash
# Fix permissions on data directory
chmod 755 data/
chmod 644 data/*.json
```

### Missing Dependencies
```bash
# Reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Older browsers may work but are not officially supported.

## Technologies Used

- **Backend**: Flask 3.0 (Python)
- **Frontend**: Bootstrap 5.3, Bootstrap Icons
- **JavaScript**: Vanilla JS (no frameworks required)
- **Data Storage**: JSON files (no database required)

## Contributing

To contribute improvements:

1. Test changes thoroughly
2. Ensure data compatibility with training pipeline
3. Update documentation as needed
4. Follow existing code style

## License

Part of the Crisis MAS LLM Training project.

## Support

For issues or questions:
- Check this README
- Review the main LLM Training documentation
- Open an issue on GitHub

## Changelog

### Version 1.0.0 (2024-11-14)
- Initial release
- Scenario management interface
- Expert profile management
- Bilingual support (Greek/English)
- Data export functionality
- API endpoints
- Sample data included
