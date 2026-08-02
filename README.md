cat > README.md << 'EOF'
# GridSecure — Electricity Theft Detection
An ML-based system to help utility and government authorities detect
patterns of electricity theft from consumption data, with a dashboard
for reviewing flagged cases.
## Team Workflow
- All modeling work happens in Google Colab / Jupyter notebooks.
- Clone this repo at the start of each Colab session:
```python
  !git clone https://github.com/Navya-Chaddha/GridSecure.git
  %cd GridSecure
```
- Large datasets and trained models are stored in a shared Google Drive
  folder — never commit them to git. Mount Drive in Colab:
```python
  from google.colab import drive
  drive.mount('/content/drive')
```
  Shared data path: `/content/drive/MyDrive/GridSecure-data/` (link: TBD)

-