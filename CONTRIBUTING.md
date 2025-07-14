# Contributing to ToxiGuard French Tweets

Thank you for your interest in contributing to ToxiGuard! This document provides guidelines and information for contributors.

## 🚀 How to Contribute

### 🐛 Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Screenshots if applicable

### 💡 Suggesting Features

1. **Check existing feature requests** first
2. **Create a detailed proposal** including:
   - Problem description
   - Proposed solution
   - Alternative solutions considered
   - Implementation ideas

### 🔧 Code Contributions

#### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ToxiGuard-French-Tweets.git
   cd ToxiGuard-French-Tweets
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If exists
   ```

4. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Workflow

1. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality

2. **Test your changes**
   ```bash
   python -m pytest tests/  # If tests exist
   python scraping.py  # Test main functionality
   ```

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Code Style Guidelines

- **Python Style**: Follow PEP 8
- **Documentation**: Use clear docstrings
- **Comments**: Explain complex logic
- **Variables**: Use descriptive names
- **Functions**: Keep them focused and small

#### Commit Message Format

Use conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code restructuring
- `test:` for adding tests
- `chore:` for maintenance tasks

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_scraping.py

# Run with coverage
python -m pytest --cov=.
```

### Writing Tests

- Add tests for new features
- Test edge cases and error conditions
- Mock external dependencies (Twitter API, etc.)
- Keep tests fast and isolated

## 📁 Project Structure

```
ToxiGuard-French-Tweets/
├── data/                   # Data storage
├── fastapi/               # API application
│   ├── model/            # ML models
│   ├── main.py           # FastAPI app
│   └── requirements.txt  # API dependencies
├── tests/                # Test files
├── docs/                 # Documentation
├── helpers.py            # Utility functions
├── model.py              # ML training
├── scraping.py           # Web scraping
└── requirements.txt      # Main dependencies
```

## 🔍 Areas for Contribution

### High Priority
- [ ] Add comprehensive tests
- [ ] Improve error handling
- [ ] Add input validation
- [ ] Performance optimization
- [ ] Documentation improvements

### Medium Priority
- [ ] Support for other languages
- [ ] Alternative ML models (BERT, etc.)
- [ ] Rate limiting improvements
- [ ] Monitoring and logging
- [ ] Configuration management

### Low Priority
- [ ] Web interface
- [ ] Batch processing API
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Performance benchmarks

## 📖 Documentation

### Updating Documentation

- Update README.md for major changes
- Add docstrings to new functions
- Update API documentation
- Include examples for new features

### Documentation Style

- Use clear, concise language
- Provide code examples
- Include visual aids when helpful
- Keep it up-to-date with code changes

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

### Communication

- Use GitHub issues for bugs and features
- Be patient and helpful in discussions
- Provide clear reproduction steps
- Share knowledge and best practices

## 🛠️ Development Environment

### Recommended Tools

- **IDE**: VS Code, PyCharm
- **Linting**: flake8, black
- **Type Checking**: mypy
- **Testing**: pytest
- **Git**: Use descriptive commit messages

### Environment Setup

```bash
# Install development tools
pip install black flake8 mypy pytest

# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

## 📝 Pull Request Process

1. **Ensure CI passes**
2. **Update documentation**
3. **Add/update tests**
4. **Request review**
5. **Address feedback**
6. **Squash commits if needed**

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests pass
- [ ] Manual testing completed
- [ ] New tests added

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🎯 Getting Help

- **Documentation**: Check README and docs/
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions
- **Code**: Read existing implementation

Thank you for contributing to ToxiGuard French Tweets! 🚀 