# Sprint 1: Execution Phase

## 1. Sprint Goal

The goal of Sprint 1 was to design and implement the core backend functionality of the TaskFlow Lite application by developing a RESTful API supporting CRUD operations for task management.



## 2. Sprint Backlog

| User Story | Description | Story Points |
| ---------- | ----------- | ------------ |
| US1        | Create task | 3            |
| US2        | View tasks  | 2            |
| US3        | Update task | 3            |
| US4        | Delete task | 2            |

**Total Story Points:** 10


## 3. Daily Scrum Summary

* **Day 1:** Project setup and Flask application initialization
* **Day 2:** Implemented task creation and retrieval endpoints
* **Day 3:** Implemented update and delete functionality
* **Day 4:** Added basic testing and debugging
* **Day 5:** Integrated components and finalized API

### Impediments:

* Encountered module import issue due to missing `__init__.py`
* Resolved by restructuring project as a Python package


## 4. Tasks Completed

* Developed REST API endpoints:

  * POST /tasks
  * GET /tasks
  * PUT /tasks/{id}
  * DELETE /tasks/{id}
* Implemented in-memory data storage
* Fixed module import issues
* Added basic test file
* Committed and pushed code to GitHub


## 5. Working Increment Delivered

A fully functional backend API was delivered with the following capabilities:

* Users can create tasks
* Users can retrieve all tasks
* Users can update task status
* Users can delete tasks

The API was successfully tested locally using a web browser and returned correct responses.


## 6. Burndown Table

| Day   | Remaining Story Points |
| ----- | ---------------------- |
| Day 1 | 10                     |
| Day 2 | 7                      |
| Day 3 | 4                      |
| Day 4 | 2                      |
| Day 5 | 0                      |


## 7. Sprint Review

The backend API was demonstrated successfully, showing all CRUD operations working as expected. The system met all Sprint 1 objectives and provided a solid foundation for further development.


## 8. Sprint Retrospective

### What went well:

* Clear backlog helped maintain focus
* Incremental development improved stability
* Early testing helped identify issues

### What could be improved:

* Better estimation of task complexity
* Earlier detection of structural issues

### Action Items for Sprint 2:

* Start testing earlier
* Improve time estimation
* Focus on UI integration and deployment

