--meta flags, jit needs to be on, 'load llvm' loads dependencies
set jit='on';
load 'llvmjit.so';

-- set jit_above_cost = 0;             --enforce jit-usage
-- set jit_inline_above_cost = 0;      --enforce jit-usage
-- set jit_optimize_above_cost = 0;    --enforce jit-usage
set jit='off';                    --enforce no jit

--drop all tables, to create new ones
drop table if exists nums;
drop table if exists nums_numeric;
drop table if exists nums_label;
drop table if exists nums_null;
drop table if exists points;
drop table if exists pages;
drop table if exists nums_matrix;
drop table if exists nums_large;

------------------------------------------create new tables and fill them with usable data----------------------------------------------
create table nums(x float not null, y float not null, z float not null, a float not null, b float not null, c float not null, d float not null);
create table nums_numeric(x float not null, y float not null, z float not null);
create table nums_label(x float not null, y float not null, z float not null);
create table nums_null(x float, y float);
create table nums_matrix(x double precision array not null, y double precision array not null, a double precision array not null, b double precision array not null);
create table points(x float not null, y float not null);
create table pages(src float not null, dst float not null, tmp_x float not null, tmp_y float not null);
create table nums_large(x1 float not null, y1 float not null, z1 float not null, x2 float not null, y2 float not null, z2 float not null, x3 float not null, y3 float not null, z3 float not null);

insert into nums select generate_series(1, 100), generate_series(101, 200), generate_series(201, 300), generate_series(1, 100), generate_series(1, 100), generate_series(1, 100), generate_series(1, 100);
insert into nums_numeric select generate_series(-2, -2), generate_series(5, 5), generate_series(12, 12);
insert into nums_label select generate_series(2, 2), generate_series(4, 4), generate_series(5, 5);
insert into nums_matrix values ('{{2,-4}, {6,8}, {2,2}}', '{{8,4,-2, 1}, {4,2,1, 1}}', '{{1,2,3,4,5}, {1,2,3,4,5}, {1,2,3,4,5}, {1,2,3,4,5}}', '{{1,2}, {1,2}, {1,2}, {1,2}, {1,2}}');
insert into nums_large select generate_series(1,10000), generate_series(10001,20000), generate_series(20001,30000), generate_series(1,10000), generate_series(10001,20000), generate_series(20001,30000), generate_series(1,10000), generate_series(10001,20000), generate_series(20001,30000);

insert into nums_null select generate_series(1, 1), generate_series(2, 2);
insert into nums_null select 1 as x, null as y;

insert into points select generate_series(1, 100), generate_series(101, 200);
insert into pages select generate_series(0.1, 1), generate_series(0.1, 1), generate_series(0.1, 1), generate_series(0.1, 1);
----------------------------------------------------------------------------------------------------------------------------------------


--------------------------------------------------------- load all functions -----------------------------------------------------------
create or replace function label(lambdacursor, "lambda") 
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/lambda_ext.so','label'
language C STRICT;

create or replace function label_fast(lambdatable, "lambda") 
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/lambda_ext.so','label_fast'
language C STRICT;

create or replace function kmeans(lambdatable, lambdatable, "lambda", int, int) 
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/kmeans_ext.so','kmeans'
language C STRICT;

create or replace function kmeans_threads(lambdatable, lambdatable, "lambda", int, int) 
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/kmeans_ext.so','kmeans_threads'
language C STRICT;

create or replace function pagerank(lambdatable, "lambda", "lambda", float, float, int, int) 
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/pagerank_ext.so','pagerank'
language C STRICT;

create or replace function pagerank_threads(lambdatable, "lambda", "lambda", float, float, int, int) 
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/pagerank_ext.so','pagerank_threads'
language C STRICT;

create or replace function autodiff_l1_2(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_l1_2'
language C STRICT;

create or replace function autodiff_l3(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_l3'
language C STRICT;

create or replace function autodiff_l4(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_l4'
language C STRICT;

create or replace function autodiff_t_l2(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_timing.so','autodiff_t_l2'
language C STRICT;

create or replace function autodiff_t_l3(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_timing.so','autodiff_t_l3'
language C STRICT;

create or replace function autodiff_t_l4(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_timing.so','autodiff_t_l4'
language C STRICT;

--parameters: inputtable(weights and data combined), lambdafunction, iterations, num attrs, batch size(if 0 or lower, BGD will be done, otherwise mini-batchGD), learning_rate
create or replace function gradient_descent_l1_2(lambdatable, "lambda", int, int, int, float)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_ext.so','gradient_descent_l1_2'
language C STRICT;

create or replace function gradient_descent_l3(lambdatable, "lambda", int, int, int, float)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_ext.so','gradient_descent_l3'
language C STRICT;

create or replace function gradient_descent_l4(lambdatable, "lambda", int, int, int, float)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_ext.so','gradient_descent_l4'
language C STRICT;

--parameters: inputtable(weights and data combined), lambdafunction, iterations, num attrs, batch size(if 0 or lower, BGD will be done, otherwise mini-batchGD), learning_rate
create or replace function gradient_descent_m_l1_2(lambdatable, "lambda", int, int, int, float)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_m_ext.so','gradient_descent_m_l1_2'
language C STRICT;

create or replace function gradient_descent_m_l3(lambdatable, "lambda", int, int, int, float)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_m_ext.so','gradient_descent_m_l3'
language C STRICT;

create or replace function gradient_descent_m_l4(lambdatable, "lambda", int, int, int, float)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_m_ext.so','gradient_descent_m_l4'
language C STRICT;
----------------------------------------------------------------------------------------------------------------------------------------


--Testing grounds of the new tupleDescriptor aquisition

-- create or replace function autodiff_debug("lambda", lambdacursor, "lambda", lambdacursor, lambdacursor)
-- returns int
-- as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_debug'
-- language C STRICT;

-- create or replace function autodiff_debug2("lambda", "lambda", lambdacursor)
-- returns int
-- as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_debug2'
-- language C STRICT;

-- create or replace function autodiff_debug3(lambdacursor, "lambda")
-- returns int
-- as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_debug3'
-- language C STRICT;

-- set jit='off';
-- select * from autodiff_debug((lambda(a)(a.x * 2)), (select * from points), (lambda(x,y)(x.y * y.z)), (select * from nums_label), (select * from nums));
-- select * from autodiff_debug2((lambda(a)(a.x * 2)), (lambda(x)(x.y * 3)), (select * from nums_label));
-- select * from autodiff_debug3((select * from points), (lambda(x)(x.y * 4)));


--test all functions and run them with their corresponding datatables
-- set jit='off'; 
-- select * from label((select x, y from nums_matrix),(lambda(a)(mat_mul(a.x,a.y)))) limit 10;
-- set jit='on'; 
-- select * from label_fast((select * from nums),(lambda(a)(atan2(a.x, a.y)))) limit 10;
-- select * from kmeans((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
-- select * from kmeans_threads((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
-- select * from pagerank((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;
-- select * from pagerank_threads((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;

-- set jit='off';
-- select * from autodiff_l1_2((select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- set jit='on';
-- select * from autodiff_l1_2((select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- select * from autodiff_l3(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- select * from autodiff_l4(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;

-- set jit='off';
-- select * from nums_matrix;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(a.x))) limit 10; 

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(silu_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(sigmoid_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(tanh_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(relu_m(a.x)))) limit 10;

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(a.x, a.y)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(relu_m(a.x), a.y)))) limit 10;

-- set jit='on';
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(a.x))) limit 10; 

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(silu_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(sigmoid_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(tanh_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(relu_m(a.x)))) limit 10;

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(a.x, a.y)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(relu_m(a.x), a.y)))) limit 10; 


-- select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(a.x))) limit 10; 

-- select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(silu_m(a.x)))) limit 10;
-- select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(sigmoid_m(a.x)))) limit 10;
-- select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(tanh_m(a.x)))) limit 10;
-- select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(relu_m(a.x)))) limit 10;

-- select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(mat_mul(a.x, a.y)))) limit 10;
-- select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(mat_mul(relu_m(a.x), a.y)))) limit 10; 


-- set jit='off';
-- select * from autodiff_l3(  (select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, relu_m(a.y)), a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y, a from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, relu_m(a.y)), a.a)))) limit 10;
-- select * from autodiff_l4(  (select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, relu_m(a.y)), mat_mul(a.x, a.y))))) limit 10;
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------Timing tests------------------------------------------------------
-- begin;
-- set jit='on';
-- \timing on
-- select * from autodiff_t_l2((select x1, y1, z1, x2, y2, z2, x3, y3, z3 from nums_large), 
--                             (lambda(x)(relu(x.x1)*(x.x3*x.x2 + x.y1*x.y3 - x.z2*x.z3)^2 + relu(x.y1)*(x.x3*x.x2 - x.y1*x.y3 + x.z2*x.z3)^2 + relu(x.z1)*(x.x3*x.x2 - x.y1*x.y3 + x.z2*x.z3)^2))) limit 1;
-- select * from autodiff_t_l2((select x1, y1, z1, x2, y2, z2, x3, y3, z3 from nums_large limit 1), 
--                             (lambda(x)(relu(x.x1)*(x.x3*x.x2 + x.y1*x.y3 - x.z2*x.z3)^2 + relu(x.y1)*(x.x3*x.x2 - x.y1*x.y3 + x.z2*x.z3)^2 + relu(x.z1)*(x.x3*x.x2 - x.y1*x.y3 + x.z2*x.z3)^2)));
-- -- select * from autodiff_t_l3(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- -- select * from autodiff_t_l4(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- commit;
----------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------- TESTS FOR GRADIENT DESCENT PURE PL/PGSQL-IMPLEMENTATIONS ----------------------------------------


-- drop table if exists data;
-- create table data (x1 float, x2 float, x3 float, x4 float, x5 float, x6 float, x7 float, x8 float, y1 float, y2 float, y3 float, y4 float, y5 float, y6 float, y7 float, y8 float, mynull float);
-- insert into data (select *,0.5+0.8*x1,0.5+0.8*x1+0.8*x2,0.5+0.8*x1+0.8*x2+0.8*x3,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6+0.8*x7,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6+0.8*x7+0.8*x8 from (select random() x1, random() x2, random() x3, random() x4, random() x5, random() x6, random() x7, random() x8, 0 from generate_series(1,10000)) as my_alias_1);

-- drop table if exists gd;
-- create table gd(id integer, a1 float, a2 float, b float);
-- insert into gd values (1, 1::float, 1::float, 1::float);

-- \timing on

-- set jit='off';

-- do $$
-- begin
--    for counter in 1..50 loop
--       insert into gd (id, a1, a2, b) 
--                      (select (id + 1)::integer as id,
--                              (a1- 0.001 * avg(d_a1))::float as a1,
--                              (a2- 0.001 * avg(d_a2))::float as a2,
--                              (b - 0.001 * avg(d_b))::float as b
--                       from autodiff_l1_2((select * from gd, (select * from data limit 10) as my_alias_2 where id = counter), 
--                                          (lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.b-x.y2)^2)))
--                       --as (id integer, a1 float, a2 float, b float, result float, d_a1 float, d_a2 float, d_b float)
--                       group by id, a1, a2, b);
--    end loop;
-- end$$;

-- with recursive gd as (
-- 	select 1, 1::float, 1::float, 1::float
-- union all
--     select (id + 1)::integer as id,
--            (a1- 0.001 * avg(d_a1))::float as a1,
--            (a2- 0.001 * avg(d_a2))::float as a2,
--            (b - 0.001 * avg(d_b))::float as b
--     from autodiff_l1_2((select * from gd, (select * from data limit 10) as my_alias_2 where id < 5), 
--                        (lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.b-x.y2)^2)))
--     group by id, a1, a2, b
-- )
-- select * from gd;

-- with recursive gd as (
-- 	select 1, 1::float, 1::float, 1::float
-- union all
--     select (id + 1)::integer as id,
--            (a1- 0.001 * avg(d_a1))::float as a1,
--            (a2- 0.001 * avg(d_a2))::float as a2,
--            (b - 0.001 * avg(d_b))::float as b
--     from autodiff_l1_2((select * from (with gd_inner as (select * from gd) select * from gd_inner, (select * from data limit 10) as my_alias_2 where id < 5)t), 
--                        (lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.b-x.y2)^2)))
--     group by id, a1, a2, b
-- )
-- select * from gd;
----------------------------------------------------------------------------------------------------------------------------------------


-------------------------------------    Testing Matrix implementations(mat mul with transpose) ----------------------------------------

--A * B (4x3 * 3x2 => 4x2)
-- drop table if exists nums_matrix_0;
-- create table nums_matrix_0(x double precision array not null, y double precision array not null);
-- insert into nums_matrix_0 values ('{{2,4,6},{2,4,6},{2,4,6},{2,4,6}}', '{{8,4}, {4,2}, {2,1}}');

-- select x, y from nums_matrix_0;
-- select * from mat_mul((select x from nums_matrix_0), (select y from nums_matrix_0));

-- --A_t * B (3x4 * 3x2 => 4x2)
-- drop table if exists nums_matrix_1;
-- create table nums_matrix_1(x double precision array not null, y double precision array not null);
-- insert into nums_matrix_1 values ('{{2,4,6,8}, {2,4,6,8}, {2,4,6,8}}', '{{8,4}, {4,2}, {2,1}}');

-- select x, y from nums_matrix_1;
-- select * from mat_mul((select x from nums_matrix_1), (select y from nums_matrix_1), 2);

-- --A * B_t (4x3 * 2x3 => 4x2)
-- drop table if exists nums_matrix_2;
-- create table nums_matrix_2(x double precision array not null, y double precision array not null);
-- insert into nums_matrix_2 values ('{{2,4,6},{2,4,6},{2,4,6},{2,4,6}}', '{{8,4,2}, {4,2,1}}');

-- select x, y from nums_matrix_2;
-- select * from mat_mul((select x from nums_matrix_2), (select y from nums_matrix_2), 1);

-- --A_t * B_t (3x4 * 2x3 => 4x2)
-- drop table if exists nums_matrix_3;
-- create table nums_matrix_3(x double precision array not null, y double precision array not null);
-- insert into nums_matrix_3 values ('{{2,4,6,8}, {2,4,6,8}, {2,4,6,8}}', '{{8,4,2}, {4,2,1}}');

-- select x, y from nums_matrix_3;
-- select * from mat_mul((select x from nums_matrix_3), (select y from nums_matrix_3), 0);

-- set jit='on';
-----------------------------------------------------------------------------------------------------------------


-------------------------------------    Testing GD operator with LinReg ----------------------------------------

-- drop table if exists data;
-- create table data (x1 float, x2 float, x3 float, x4 float, x5 float, x6 float, x7 float, x8 float, y1 float, y2 float, y3 float, y4 float, y5 float, y6 float, y7 float, y8 float, mynull float);
-- insert into data (select *,0.5+0.8*x1,0.5+0.8*x1+0.8*x2,0.5+0.8*x1+0.8*x2+0.8*x3,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6+0.8*x7,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6+0.8*x7+0.8*x8 from (select random() x1, random() x2, random() x3, random() x4, random() x5, random() x6, random() x7, random() x8, 0 from generate_series(1,10000)) as my_alias_1);

-- drop table if exists gd;
-- create table gd(a1 float, a2 float,a3 float, a4 float, a5 float, a6 float, a7 float, a8 float, b float);
-- insert into gd values (10::float, 10::float, 10::float, 10::float, 10::float, 10::float, 10::float, 10::float, 10::float);

-- set jit='off';
--params for gd: 1.Number of iterations/epochs  2.number of attributes  3.batch_size(-1 means whole data_set as one batch)  4.learning_rate
-- select *
-- from gradient_descent_l1_2((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       100, 9, -1, 0.001);

-- select autodiff_l1_2.result
-- from autodiff_l1_2((select * from gd, (select * from data limit 1) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;

-- set jit='on';

-- select *
-- from gradient_descent_l1_2((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       100, 9, -1, 0.001);

-- select autodiff_l1_2.result
-- from autodiff_l1_2((select * from gd, (select * from data limit 1) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;

-- select *
-- from gradient_descent_l3((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       100, 9, -1, 0.001);

-- select autodiff_l3.result
-- from autodiff_l3((select * from gd, (select * from data limit 1) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;

-- select *
-- from gradient_descent_l4((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       100, 9, -1, 0.001);

-- select autodiff_l4.result
-- from autodiff_l4((select * from gd, (select * from data limit 1) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;

----------------------------------------------------------------------------------------------------------------------


-------------------------------------    Testing GD operator with neural nets ----------------------------------------
drop table if exists nn_table;
drop table if exists iris;
drop table if exists iris3;

create table if not exists iris (sepal_length float,sepal_width float,petal_length float,petal_width float,species int);
create table if not exists iris3 (img float[], one_hot float[]);

copy iris from '/home/clemens/masterarbeit/psql-autodiff/src/ext/iris.csv' DELIMITER ',' CSV HEADER;
insert into iris3 (select array[[sepal_length/10,sepal_width/10,petal_length/10,petal_width/10]] as img, 
                    array[(array_fill(0::float,array[species]) || 1::float ) || array_fill(0::float,array[2-species])] as one_hot from iris limit 150);

create table nn_table(w_xh float array not null, w_ho float array not null);
insert into nn_table select
    (select array_agg(array_agg) from generate_series(1,4), (select array_agg(random()) from generate_series(1,20)) as foo),
    (select array_agg(array_agg) from generate_series(1,20), (select array_agg(random()) from generate_series(1,3)) as foo);

-- select * from nn_table, iris3 tablesample bernoulli (1);

--params for gd: 1.Number of iterations/epochs  2.number of attributes  3.batch_size(-1 means whole data_set as one batch)  4.learning_rate
select *
from gradient_descent_m_l1_2((select * from nn_table, iris3 tablesample bernoulli (1)), 
                             (lambda(x)(softmax_cce(tanh_m(x.img**x.w_xh)**x.w_ho, x.one_hot))),
                             1, 2, -1, 0.001);

-- with gd(id, w_xh, w_ho) as (
--     select w_xh, w_ho
--     from autodiff_l1_2((select * from iris3 tablesample bernoulli (1), nn_table), (lambda(x)(softmax_cce(tanh_m(x.img**x.w_xh)**x.w_ho, x.one_hot))))
-- ), test as (select id, correct, count(*) as count from (select id, index_max(softmax(tanh_m(img**w_xh)**w_ho))=index_max(one_hot) as correct from iris3, gd) as foo group by id, correct)
-- select id, count*1.0/(select sum(count) from test t2 where t1.id=t2.id) from test t1 where correct=true order by id; 
----------------------------------------------------------------------------------------------------------------------

-------------------------------------    Testing pure SQL Neural Net implementations ----------------------------------------


-- select array_agg(array_agg) from generate_series(1,4), (select array_agg(random()) from generate_series(1,20)) as foo;
-- select array_agg(array_agg) from generate_series(1,20), (select array_agg(random()) from generate_series(1,3)) as foo;
-- with nn_table(id, w_xh, w_ho) as (
--     select 0,
--     (select array_agg(array_agg) from generate_series(1,4), (select array_agg(random()) from generate_series(1,20)) as foo),
--     (select array_agg(array_agg) from generate_series(1,20), (select array_agg(random()) from generate_series(1,3)) as foo)
-- ) select * from nn_table;

-- create aggregate avg(double precision[])
-- (
--     sfunc = mat_avg,
--     stype = float8[],
--     finalfunc = mat_avg_final,
--     initcond = '{0.0, 0}'
-- );

-- drop table if exists nn_table;
-- drop table if exists iris;
-- drop table if exists iris3;

-- create table if not exists iris (sepal_length float,sepal_width float,petal_length float,petal_width float,species int);
-- create table if not exists iris3 (img float[], one_hot float[]);

-- copy iris from '/home/clemens/masterarbeit/psql-autodiff/src/ext/iris.csv' DELIMITER ',' CSV HEADER;
-- insert into iris3 (select array[[sepal_length/10,sepal_width/10,petal_length/10,petal_width/10]] as img, 
--                     array[(array_fill(0::float,array[species]) || 1::float ) || array_fill(0::float,array[2-species])] as one_hot from iris limit 150);

-- create table nn_table(id int not null, w_xh float array not null, w_ho float array not null);
-- insert into nn_table select 0,
--     (select array_agg(array_agg) from generate_series(1,4), (select array_agg(random()) from generate_series(1,20)) as foo),
--     (select array_agg(array_agg) from generate_series(1,20), (select array_agg(random()) from generate_series(1,3)) as foo);

-- with gd(id, w_xh, w_ho) as (
--     select id+1 as id, w_xh - 0.2 * (transpose(img)**d_xh) as w_xh, w_ho - 0.2 * (transpose(a_xh)**d_ho) as w_ho
--         from (
--             select l_xh *(1 - (a_xh * a_xh)) as d_xh, *
--             from (
--                 select d_ho**transpose(w_ho) as l_xh, *
--                 from (
--                     select l_ho * (1 - (a_ho * a_ho)) as d_ho, *
--                     from (
--                         select 2*(a_ho-one_hot) as l_ho, *
--                         from (
--                             select tanh_m(a_xh**w_ho) as a_ho, *
--                             from (
--                                 select tanh_m(img**w_xh) as a_xh, *
--                                 from (select * from iris3 tablesample bernoulli (1)) as foo7, nn_table) as foo6
--                         ) as foo5
--                     ) as foo4
--                 ) as foo3
--             ) as foo2
--         ) as foo1
-- ), test as (select id, correct, count(*) as count from (select id, index_max(softmax(tanh_m(img**w_xh)**w_ho))=index_max(one_hot) as correct from iris3, gd) as foo group by id, correct)
-- select id, count*1.0/(select sum(count) from test t2 where t1.id=t2.id) from test t1 where correct=true order by id;
-- begin;

-- with gd(id, w_xh, w_ho) as (
--     select id+1 as id, w_xh - 0.2 * (d_w_xh) as w_xh, w_ho - 0.2 * (d_w_ho) as w_ho
--     from autodiff_l1_2((select * from iris3 tablesample bernoulli (1), nn_table), (lambda(x)(softmax_cce(tanh_m(x.img**x.w_xh)**x.w_ho, x.one_hot))))
-- ), test as (select id, correct, count(*) as count from (select id, index_max(softmax(tanh_m(img**w_xh)**w_ho))=index_max(one_hot) as correct from iris3, gd) as foo group by id, correct)
-- select id, count*1.0/(select sum(count) from test t2 where t1.id=t2.id) from test t1 where correct=true order by id; 

-- set jit='on';
-- with gd(id, w_xh, w_ho) as (
--     select id+1 as id, w_xh - 0.2 * (d_w_xh) as w_xh, w_ho - 0.2 * (d_w_ho) as w_ho
--     from autodiff_l1_2((select * from iris3 tablesample bernoulli (1), nn_table), (lambda(x)(softmax_cce(tanh_m(x.img**x.w_xh)**x.w_ho, x.one_hot))))
-- ), test as (select id, correct, count(*) as count from (select id, index_max(softmax(tanh_m(img**w_xh)**w_ho))=index_max(one_hot) as correct from iris3, gd) as foo group by id, correct)
-- select id, count*1.0/(select sum(count) from test t2 where t1.id=t2.id) from test t1 where correct=true order by id;

-- with gd(id, w_xh, w_ho) as (
--     select id+1 as id, w_xh - 0.2 * (d_w_xh) as w_xh, w_ho - 0.2 * (d_w_ho) as w_ho
--     from autodiff_l3((select * from iris3 tablesample bernoulli (1), nn_table), (lambda(x)(softmax_cce(tanh_m(x.img**x.w_xh)**x.w_ho, x.one_hot))))
-- ), test as (select id, correct, count(*) as count from (select id, index_max(softmax(tanh_m(img**w_xh)**w_ho))=index_max(one_hot) as correct from iris3, gd) as foo group by id, correct)
-- select id, count*1.0/(select sum(count) from test t2 where t1.id=t2.id) from test t1 where correct=true order by id;

-- commit;

-- \echo "hello"


-- select avg(img) as img_avg, avg(one_hot) as one_hot_avg from iris3 limit 5;


-- select id+1 as id, w_xh - 0.2 * avg(transpose(img)**d_xh) as w_xh, w_ho as w_ho
-- from (
--     select w_ho - 0.2 * avg(transpose(a_xh)**d_ho) as w_ho, id, w_xh, img, d_xh
--     from (
--         select l_xh *(1 - (a_xh * a_xh)) as d_xh, *
--         from (
--             select d_ho**transpose(w_ho) as l_xh, *
--             from (
--                 select l_ho * (1 - (a_ho * a_ho)) as d_ho, *
--                 from (
--                     select 2*(a_ho-one_hot) as l_ho, *
--                     from (
--                         select tanh_m(a_xh**w_ho) as a_ho, *
--                         from (
--                             select tanh_m(img**w_xh) as a_xh, *
--                             from (select * from iris3 limit 10) as foo7, nn_table) as foo6
--                     ) as foo5
--                 ) as foo4
--             ) as foo3
--         ) as foo2
--     ) as foo1 group by id, w_xh, w_ho, img, d_xh
-- ) as foo0
-- group by id, w_xh, w_ho;

-- select distinct array_dims(tanh_m(img**w_xh)) as a_xh
--                         from (select * from iris3 limit 10) as foo7, nn_table;

-- select distinct array_dims(tanh_m(a_xh**w_ho)) as a_ho
--                     from (
--                         select tanh_m(img**w_xh) as a_xh, *
--                         from (select * from iris3 limit 10) as foo7, nn_table) as foo6;

-- select distinct array_dims(2*(a_ho-one_hot)) as l_ho
--                 from (
--                     select tanh_m(a_xh**w_ho) as a_ho, *
--                     from (
--                         select tanh_m(img**w_xh) as a_xh, *
--                         from (select * from iris3 limit 10) as foo7, nn_table) as foo6
--                 ) as foo5;

-- select distinct array_dims(l_ho * (1 - (a_ho * a_ho))) as d_ho
--             from (
--                 select 2*(a_ho-one_hot) as l_ho, *
--                 from (
--                     select tanh_m(a_xh**w_ho) as a_ho, *
--                     from (
--                         select tanh_m(img**w_xh) as a_xh, *
--                         from (select * from iris3 limit 10) as foo7, nn_table) as foo6
--                 ) as foo5
--             ) as foo4;

-- select distinct array_dims(d_ho**transpose(w_ho)) as l_xh
--         from (
--             select l_ho * (1 - (a_ho * a_ho)) as d_ho, *
--             from (
--                 select 2*(a_ho-one_hot) as l_ho, *
--                 from (
--                     select tanh_m(a_xh**w_ho) as a_ho, *
--                     from (
--                         select tanh_m(img**w_xh) as a_xh, *
--                         from (select * from iris3 limit 10) as foo7, nn_table) as foo6
--                 ) as foo5
--             ) as foo4
--         ) as foo3;

-- select distinct array_dims(l_xh *(1 - (a_xh * a_xh))) as d_xh
--     from (
--         select d_ho**transpose(w_ho) as l_xh, *
--         from (
--             select l_ho * (1 - (a_ho * a_ho)) as d_ho, *
--             from (
--                 select 2*(a_ho-one_hot) as l_ho, *
--                 from (
--                     select tanh_m(a_xh**w_ho) as a_ho, *
--                     from (
--                         select tanh_m(img**w_xh) as a_xh, *
--                         from (select * from iris3 limit 10) as foo7, nn_table) as foo6
--                 ) as foo5
--             ) as foo4
--         ) as foo3
--     ) as foo2;

-----------------------------------------------------------------------------------------------


----------------------------------- Schachtelungen Versuch ------------------------------------ (Fail: cache lookup failed for type 0(oder auch, OID 0 nicht vergeben))
-- drop table if exists gd;
-- create table if not exists gd(id integer, a1 float, b float);
-- insert into gd values(1::integer, 1::float, 1::float);

-- drop table if exists data;
-- create table if not exists data(x1 float, x2 float, y1 float, y2 float, table_delimiter float);
-- insert into data(select *, 0.5+0.8*x1, 0.5+0.8*x1+0.8*x2 from (select random() x1, random() x2, 0 from generate_series(1, 10)) as pg_alias);

-- set jit='off';

-- \set verbosity verbose

-- select  (id+1)::integer as id,
-- 		(a1 - 0.001*avg(d_a1))::float as a1,
-- 		(b  - 0.001*avg(d_b))::float as b
-- from autodiff_l1_2
-- (
--     (
--         select  (id+1)::integer as id,
-- 		        (a1 - 0.001*avg(d_a1))::float as a1,
-- 		        (b  - 0.001*avg(d_b))::float as b,
--                 (x1)::float as x1, 
--                 (x2)::float as x2, 
--                 (y1)::float as y1
--         from autodiff_l1_2(
--             (
--                 select * from gd, (select x1, x2, y1 from data limit 10) as pg_alias where id <= 1
--             ),
--         (lambda(x)(( x.a1*x.x1 + x.b-x.y1)^2)))
--         group by id, a1, b, x1, x2, y1
--     ),
--     (lambda(x)(( x.a1*x.x1 + x.b-x.y1)^2))
-- )
-- group by id, a1, b;
-----------------------------------------------------------------------------------------------

