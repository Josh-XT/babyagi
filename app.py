from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from TMS import TaskManagementSystem
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
tms = TaskManagementSystem()

tms_thread = None

def run_tms():
    tms.run()

def print_output(message, event_name):
    tms.task_list.append(message)
    socketio.emit(event_name, list(tms.task_list))

    print('print_output called with event_name:', event_name, 'and message:', message)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    task_list = list(tms.task_list)
    emit('task_list_update', task_list)

    print('task_list_update emitted with task_list:', task_list)


@socketio.on('set_objective')
def handle_set_objective(objective):
    global tms_thread

    if not tms_thread:
        tms.primary_objective = objective
        tms_thread = threading.Thread(target=run_tms)
        tms_thread.daemon = True
        tms_thread.start()

    print('Objective set to:', objective)

if __name__ == '__main__':
    socketio.run(app, debug=True)
