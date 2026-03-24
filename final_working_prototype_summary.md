# Final Working Prototype Summary

## 1. Repository Link

https://github.com/Damas200/taskflow-lite



## 2. How to Run the Application

### Local Execution

```bash
python -m app.app
```

Access:
http://127.0.0.1:5001



### Docker Execution

```bash
docker build -t taskflow-lite .
docker run -p 5001:5001 taskflow-lite
```



## 3. Key Features

* Create tasks
* View tasks
* Update tasks
* Delete tasks
* Interactive UI
* REST API backend



## 4. Working System Description

The application allows users to interact with a simple web interface to manage tasks. The frontend communicates with the backend API to perform CRUD operations, and the system functions correctly in both local and containerized environments.

