from flask import Flask, render_template, request
import numpy as np
from scipy.linalg import lu_factor,lu_solve

app = Flask(__name__)

input_size = 2

@app.route('/')
def input():
    return render_template('input.html')

@app.route('/index',methods=['POST'])
def index():
    size = int(request.form['size'])
    global input_size
    input_size = size
    print(input_size)
    return render_template('index.html',size=size)

@app.route('/result', methods=['POST'])
def result():
    A = np.zeros((input_size, input_size))
    B = np.zeros(input_size)
    for i in range(input_size):
        for j in range(input_size):
            A[i, j] = float(request.form['matrix{}{}'.format(i, j)])
        B[i] = float(request.form['vector{}'.format(i)])

    lu, piv = lu_factor(A)
    l_matrix = np.tril(lu, k=-1) + np.eye(len(A))
    u_matrix = np.triu(lu)
    p_matrix = np.eye(len(A))[:, piv]
    x = lu_solve((lu, piv), B)
    
    return render_template('result.html',l=l_matrix,p=p_matrix,u=u_matrix, x=x,c=1)

if __name__ == '__main__':
    app.run(debug=True)
