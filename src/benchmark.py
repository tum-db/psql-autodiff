import psycopg2
from datetime import datetime
from statistics import median

exp_test_n = 7
total_tests = 5
lambda_func = "sin(a.x) / cos(a.y) + sqrt(a.z)"
print_as_plot = False
sql_setup = ""

connection = psycopg2.connect(user="clemens",
                                  password="",
                                  host="localhost",
                                  port="5432",
                                  database="test")
cursor = connection.cursor()
cursor.execute("load 'llvmjit.so'")
	
def measure(q, c):
    result = []
    for n in range(1, total_tests):
        start = datetime.now()
        c.execute(q)
        result.append((datetime.now() - start).total_seconds() * 1000.0)
    return median(result)

def exp_test(q, e, l):
    if print_as_plot:
        print("\\addplot coordinates {", end='')
    else:
        print("\nPostgres query: " + q.format("<SIZE>", lambda_func))

    for n in range(1, e+1):
        r = measure(q.format(n, lambda_func), cursor)
        if print_as_plot:
            print("({}, {:.3f})".format(10**n, r), end='')
        else:
            print("Input size: 10^{} - AutoDiff-L{}: {:.3f}".format(n, l, r))

    if print_as_plot:
        print("}};\n\\addlegendentry{{L{}}};".format(l))


print("\n==== AutoDiff: L1-4 Test ====")
cursor.execute("set jit='off'")
exp_test("select * from autodiff_l1_2((select x, y, z from perftests{}), (lambda(a)({})))", exp_test_n, 1)
cursor.execute("set jit='on'")
cursor.execute("load 'llvmjit.so'")
cursor.execute("set jit_above_cost = 0")
cursor.execute("set jit_inline_above_cost = 0")
cursor.execute("set jit_optimize_above_cost = 0")
exp_test("select * from autodiff_l1_2((select x, y, z from perftests{}), (lambda(a)({})))", exp_test_n, 2)
exp_test("select * from autodiff_l3((select x, y, z from perftests{}), (lambda(a)({})))", exp_test_n, 3)
exp_test("select * from autodiff_l4((select x, y, z from perftests{}), (lambda(a)({})))", exp_test_n, 4)
