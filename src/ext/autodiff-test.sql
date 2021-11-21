--meta flags, jit needs to be on, 'load llvm' loads dependencies
set jit='on';
load 'llvmjit.so';
set autocommit=0;

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

--create new tables and fill them with usable data
create table nums(x float not null, y float not null, z float not null, a float not null, b float not null, c float not null, d float not null);
create table nums_numeric(x float not null, y float not null, z float not null);
create table nums_label(x float not null, y float not null, z float not null);
create table nums_null(x float, y float);
create table nums_matrix(x double precision array not null, y double precision array not null);
create table points(x float not null, y float not null);
create table pages(src float not null, dst float not null, tmp_x float not null, tmp_y float not null);

insert into nums select generate_series(1, 100), generate_series(101, 200), generate_series(201, 300), generate_series(1, 100), generate_series(1, 100), generate_series(1, 100), generate_series(1, 100);
insert into nums_numeric select generate_series(-2, -2), generate_series(5, 5), generate_series(12, 12);
insert into nums_label select generate_series(2, 2), generate_series(4, 4), generate_series(5, 5);
insert into nums_matrix values ('{{2,-4}, {6,8}, {2,2}}', '{{8,4,-2}, {4,2,1}}');

insert into nums_null select generate_series(1, 1), generate_series(2, 2);
insert into nums_null select 1 as x, null as y;

insert into points select generate_series(1, 100), generate_series(101, 200);
insert into pages select generate_series(0.1, 1), generate_series(0.1, 1), generate_series(0.1, 1), generate_series(0.1, 1);


--load all functions
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

create or replace function autodiff_debug("lambda", lambdacursor, "lambda", lambdacursor, lambdacursor)
returns int
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_debug'
language C STRICT;

create or replace function autodiff_t_l1_2(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_timing.so','autodiff_t_l1_2'
language C STRICT;

create or replace function autodiff_t_l3(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_timing.so','autodiff_t_l3'
language C STRICT;

create or replace function autodiff_t_l4(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_timing.so','autodiff_t_l4'
language C STRICT;

create or replace function gradient_descent_l1_2(lambdatable, "lambda", int, int)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_ext.so','gradient_descent_l1_2'
language C STRICT;

create or replace function gradient_descent_l3(lambdatable, "lambda", int, int)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_ext.so','gradient_descent_l3'
language C STRICT;

create or replace function gradient_descent_l4(lambdatable, "lambda", int, int)
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/gradient_desc_ext.so','gradient_descent_l4'
language C STRICT;

set jit='off';
--select * from autodiff_debug((lambda(a)(a.x * 2)), (select * from points), (lambda(x,y)(x.y * y.z)), (select * from nums_label), (select * from nums));


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

set jit='off';
select * from nums_matrix;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(a.x))) limit 10; 

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(silu_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(sigmoid_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(tanh_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(relu_m(a.x)))) limit 10;

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(a.x, a.y)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(relu_m(a.x), a.y)))) limit 10;

set jit='on';
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(a.x))) limit 10; 

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(silu_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(sigmoid_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(tanh_m(a.x)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(relu_m(a.x)))) limit 10;

-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(a.x, a.y)))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(relu_m(a.x), a.y)))) limit 10; 


select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(a.x))) limit 10; 

select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(silu_m(a.x)))) limit 10;
select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(sigmoid_m(a.x)))) limit 10;
select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(tanh_m(a.x)))) limit 10;
select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(relu_m(a.x)))) limit 10;

select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(mat_mul(a.x, a.y)))) limit 10;
select * from autodiff_l3((select x, y from nums_matrix), (lambda(a)(mat_mul(relu_m(a.x), a.y)))) limit 10; 


-- set jit='on';
-- select * from autodiff_l3(  (select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, relu_m(a.y)), mat_mul(a.x, a.y))))) limit 10;
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, relu_m(a.y)), mat_mul(a.x, a.y))))) limit 10;
-- select * from autodiff_l4(  (select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, relu_m(a.y)), mat_mul(a.x, a.y))))) limit 10;

-- --Timing tests for comp vs exec
-- set jit='off';
-- select * from autodiff_t_l1_2((select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- set jit='on';
-- select * from autodiff_t_l1_2((select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- select * from autodiff_t_l3(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- select * from autodiff_t_l4(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;




-- TESTS FOR GRADIENT DESCENT PURE PL/PGSQL-IMPLEMENTATIONS


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

-- drop table if exists data;
-- create table data (x1 float, x2 float, x3 float, x4 float, x5 float, x6 float, x7 float, x8 float, y1 float, y2 float, y3 float, y4 float, y5 float, y6 float, y7 float, y8 float, mynull float);
-- insert into data (select *,0.5+0.8*x1,0.5+0.8*x1+0.8*x2,0.5+0.8*x1+0.8*x2+0.8*x3,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6+0.8*x7,0.5+0.8*x1+0.8*x2+0.8*x3+0.8*x4+0.8*x5+0.8*x6+0.8*x7+0.8*x8 from (select random() x1, random() x2, random() x3, random() x4, random() x5, random() x6, random() x7, random() x8, 0 from generate_series(1,10000)) as my_alias_1);

-- drop table if exists gd;
-- create table gd(a1 float, a2 float,a3 float, a4 float, a5 float, a6 float, a7 float, a8 float, b float);
-- insert into gd values (10::float, 10::float, 10::float, 10::float, 10::float, 10::float, 10::float, 10::float, 10::float);

-- set jit='off';

-- select *
-- from gradient_descent_l1_2((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       50, 8);

-- select d_a1, d_a2, d_a3
-- from autodiff_l1_2((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;

-- set jit='on';

-- select *
-- from gradient_descent_l1_2((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       50, 8);

-- select d_a1, d_a2, d_a3
-- from autodiff_l1_2((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;

-- select *
-- from gradient_descent_l3((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       50, 8);

-- select d_a1, d_a2, d_a3
-- from autodiff_l3((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;

-- select *
-- from gradient_descent_l4((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2), 
--                       50, 8);

-- select d_a1, d_a2, d_a3
-- from autodiff_l4((select * from gd, (select * from data limit 1000) as pg_alias), 
--                       lambda(x)((x.a1*x.x1 + x.a2*x.x2 + x.a3*x.x3 + x.a4*x.x4 + x.a5*x.x5 + x.a6*x.x6 + x.a7*x.x7 + x.a8*x.x8 + x.b - x.y8)^2)) limit 1;
