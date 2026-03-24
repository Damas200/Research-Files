# DevOps Practices Implemented

## 1. Version Control

Git was used to manage source code, track changes, and maintain version history. GitHub was used as a remote repository to enable continuous integration and collaboration.

---

## 2. Continuous Integration (CI)

A CI pipeline was configured using GitHub Actions. Each push triggered:

* Installation of dependencies
* Execution of test scripts
* Validation of code functionality

This ensured early detection of errors and maintained code quality.

---

## 3. Continuous Deployment (CD)

Although full cloud deployment was not implemented, the application was prepared for deployment through Docker containerization, enabling consistent execution across environments.

---

## 4. Containerization

Docker was used to package the application into a container. This ensured:

* Environment consistency
* Portability across systems
* Simplified deployment

---

## 5. Automated Testing

Basic unit tests were implemented using pytest to verify functionality and support CI validation.

---

## 6. Logging and Monitoring

Flask debug logs were used during development to monitor application behavior, detect errors, and support debugging.

---

## 7. DevOps Impact on Agile Delivery

DevOps practices significantly improved Agile delivery by:

* Enabling faster feedback through CI
* Reducing integration issues
* Automating repetitive tasks
* Ensuring reliable deployments

---
