--meta flags, jit needs to be on, 'load llvm' loads dependencies
set jit='on';
load 'llvmjit.so';

-- set jit_above_cost = 0;             --enforce jit-usage
-- set jit_inline_above_cost = 0;      --enforce jit-usage
-- set jit_optimize_above_cost = 0;    --enforce jit-usage
set jit='off';                    --enforce no jit

--drop all tables, to create new ones
drop table if exists nums_numeric;
drop table if exists nums_matrix;

--create new tables and fill them with usable data
create table nums_numeric(x float not null, y float not null, z float not null);
create table nums_matrix(x double precision array not null, y double precision array not null);

insert into nums_numeric select generate_series(-2, -2), generate_series(5, 5), generate_series(12, 12);
insert into nums_matrix values ('{{2,4}, {6,8}, {2,2}}', '{{8,4,2}, {4,2,1}}');



--load all functions
create or replace function autodiff_l1_2(lambdacursor, "lambda")
returns setof record
as '/psql/src/ext/autodiff_ext.so','autodiff_l1_2'
language C STRICT;

create or replace function autodiff_l3(lambdacursor, "lambda")
returns setof record
as '/psql/src/ext/autodiff_ext.so','autodiff_l3'
language C STRICT;

create or replace function autodiff_l4(lambdacursor, "lambda")
returns setof record
as '/psql/src/ext/autodiff_ext.so','autodiff_l4'
language C STRICT;

create or replace function gradient_descent_l1_2(lambdatable, "lambda", int, int)
returns setof record
as '/psql/src/ext/gradient_desc_ext.so','gradient_descent_l1_2'
language C STRICT;

create or replace function gradient_descent_l3(lambdatable, "lambda", int, int)
returns setof record
as '/psql/src/ext/gradient_desc_ext.so','gradient_descent_l3'
language C STRICT;

create or replace function gradient_descent_l4(lambdatable, "lambda", int, int)
returns setof record
as '/psql/src/ext/gradient_desc_ext.so','gradient_descent_l4'
language C STRICT;

-- -- test all functions and run them with their corresponding datatables
-- set jit='off';
-- select * from autodiff_l1_2((select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- set jit='on';
-- select * from autodiff_l1_2((select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- select * from autodiff_l3(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
-- select * from autodiff_l4(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;

-- set jit='off';
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, a.y), mat_mul(a.x, a.y))))) limit 10;
-- set jit='on';
-- select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, a.y), mat_mul(a.x, a.y))))) limit 10;
-- select * from autodiff_l3(  (select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, a.y), mat_mul(a.x, a.y))))) limit 10;
-- select * from autodiff_l4(  (select x, y from nums_matrix), (lambda(a)(mat_mul(mat_mul(a.x, a.y), mat_mul(a.x, a.y))))) limit 10;

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
