--meta flags, jit needs to be on, 'load llvm' loads dependencies
set jit='on';
load 'llvmjit.so';

set jit_above_cost = 0;             --enforce jit-usage
set jit_inline_above_cost = 0;      --enforce jit-usage
set jit_optimize_above_cost = 0;    --enforce jit-usage

--set jit='off';                    --enforce no jit

--drop all tables, to create new ones
drop table if exists nums;
drop table if exists nums_numeric;
drop table if exists nums_label;
drop table if exists nums_null;
drop table if exists points;
drop table if exists pages;
drop table if exists nums_matrix;

--create new tables and fill them with usable data
create table nums(x float not null, y float not null, z float not null, a float not null, b float not null, c float not null);
create table nums_numeric(x float not null, y float not null, z float not null);
create table nums_label(x float not null, y float not null, z float not null);
create table nums_null(x float, y float);
create table points(x float not null, y float not null);
create table pages(src float not null, dst float not null);
create table nums_matrix(x double precision array not null, y double precision array not null);

insert into nums select generate_series(1, 100), generate_series(101, 200), generate_series(201, 300), generate_series(1, 100), generate_series(1, 100), generate_series(1, 100);
insert into nums_numeric select generate_series(-2, -2), generate_series(5, 5), generate_series(12, 12);
insert into nums_label select generate_series(2, 2), generate_series(4, 4), generate_series(5, 5);

insert into nums_null select generate_series(1, 1), generate_series(2, 2);
insert into nums_null select 1 as x, null as y;

insert into points select generate_series(1, 100), generate_series(101, 200);
insert into pages select generate_series(0.1, 1), generate_series(0.1, 1);

insert into nums_matrix values ('{{2,4}, {6,8}}', '{{8,4}, {4,2}}');

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

--test all functions and run them with their corresponding datatables
--set jit='off'; 
--select * from label((select x, y from nums_matrix),(lambda(a)(mat_mul(a.x,a.y)))) limit 10;
--select * from label_fast((select * from nums),(lambda(a)(atan2(a.x, a.y)))) limit 10;
--select * from kmeans((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
--select * from kmeans_threads((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
--select * from pagerank((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;
--select * from pagerank_threads((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;

--select * from autodiff_l1_2((select x, y, z from nums),(lambda(a)(a.x*a.x + 2 * a.y - a.z))) limit 10;
--select * from autodiff_l1_2((select x, y from nums),(lambda(a)(-atan2(a.x, a.y)))) limit 10;
--select * from autodiff_l1_2((select x, y from nums_null), (lambda(a)(a.x + a.y))) limit 10;
--select * from autodiff_l1_2((select x, y, z from nums),lambda(a)((a.x*a.y) + a.z/2)); 

--set jit='off';
--select * from autodiff_l1_2((select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
--set jit='on';
--select * from autodiff_l3(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
--select * from autodiff_l4(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;

set jit='off';
select * from autodiff_l1_2((select x, y from nums_matrix), (lambda(a)(mat_mul(a.x, a.y)))) limit 10;
--set jit='on';
--select * from autodiff_l3(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;
--select * from autodiff_l4(  (select x, y, z from nums_numeric), (lambda(a)(relu(a.x) + relu(a.y) + relu(a.z)))) limit 10;










-- --A * B (4x3 * 3x2 => 4x2)
-- drop table if exists nums_matrix_0;
-- create table nums_matrix_0(x double precision array not null, y double precision array not null);
-- insert into nums_matrix_0 values ('{{2,4,6},{2,4,6},{2,4,6},{2,4,6}}', '{{8,4}, {4,2}, {2,1}}');

-- select x, y from nums_matrix_0;
-- select * from mat_mul((select x from nums_matrix_0), (select y from nums_matrix_0), 3);

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