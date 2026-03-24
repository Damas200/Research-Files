# Sprint 2: Execution Phase

## 1. Sprint Goal

The goal of Sprint 2 was to enhance the TaskFlow Lite application by developing a user interface, integrating frontend and backend components, and implementing DevOps practices such as containerization and continuous integration.


## 2. Sprint Backlog

| User Story | Description                 | Story Points |
| ---------- | --------------------------- | ------------ |
| US5        | Build basic UI              | 3            |
| US6        | Integrate frontend with API | 3            |
| US7        | Dockerize application       | 2            |
| US8        | Improve CI/CD pipeline      | 2            |

**Total Story Points:** 10


## 3. Work Completed

* Developed a simple HTML-based user interface
* Connected frontend to backend using JavaScript fetch API
* Enabled dynamic task creation and display
* Implemented Docker containerization
* Configured Flask to run on `0.0.0.0` for container access
* Resolved Docker networking issues (WSL environment)
* Handled port conflicts during execution


## 4. DevOps in Action

### Continuous Integration

* Code pushed to GitHub triggered CI pipeline
* Dependencies installed automatically
* Basic tests executed

### Containerization

* Docker used to package application
* Ensured consistent runtime environment

### Deployment

* Application successfully executed inside Docker container
* Verified through logs and local testing


## 5. Monitoring and Logging

* Flask logs used to monitor requests and debug errors
* Identified and resolved runtime and networking issues


## 6. Working Increment Delivered

The final system includes:

* Fully functional web interface
* Backend API handling CRUD operations
* Real-time interaction between frontend and backend
* Application running both locally and in Docker


## 7. Sprint Review

The completed system was demonstrated successfully, showing full functionality from user input to backend processing and output display. All Sprint 2 objectives were achieved.


## 8. Sprint Retrospective

### What went well:

* Smooth frontend-backend integration
* Successful Docker implementation
* Improved development efficiency

### Challenges faced:

* Docker networking issues in WSL environment
* Port conflicts during execution

### Improvements:

* Better understanding of container networking
* Enhanced debugging skills
* Improved planning accuracy


## 9. Velocity Improvement

Sprint 2 showed improved velocity due to:

* Familiarity with tools and codebase
* Better estimation and planning
* Reduced setup time

