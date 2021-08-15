--meta flags, jit needs to be on, load llvm loads some dependencies
set jit='on';
load 'llvmjit.so';

set jit_above_cost = 0;             --enforce jit-usage
set jit_inline_above_cost = 0;      --enforce jit-usage
set jit_optimize_above_cost = 0;    --enforce jit-usage

--set jit_above_cost = 100000000000;             --enforce no jit
--set jit_inline_above_cost = 100000000000;      --enforce no jit
--set jit_optimize_above_cost = 100000000000;    --enforce no jit

--drop all tables, to create new and fresh ones
drop table if exists nums;
drop table if exists nums_numeric;
drop table if exists nums_label;
drop table if exists nums_null;
drop table if exists points;
drop table if exists pages;


--create new tables and fill them with usable data
create table nums(x float not null, y float not null, z float not null, a float not null, b float not null, c float not null);
create table nums_numeric(x numeric not null, y numeric not null);
create table nums_label(x float not null, y float not null, z float not null);
create table nums_null(x float, y float);
create table points(x float not null, y float not null);
create table pages(src float not null, dst float not null);

insert into nums select generate_series(1, 100), generate_series(101, 200), generate_series(201, 300), generate_series(1, 100), generate_series(1, 100), generate_series(1, 100);
insert into nums_numeric select generate_series(2, 2), generate_series(64, 64);
insert into nums_label select generate_series(2, 2), generate_series(4, 4), generate_series(5, 5);

insert into nums_null select generate_series(1, 1), generate_series(2, 2);
insert into nums_null select 1 as x, null as y;

insert into points select generate_series(1, 100), generate_series(101, 200);
insert into pages select generate_series(0.1, 1), generate_series(0.1, 1);


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

create or replace function autodiff(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff'
language C STRICT;

create or replace function autodiff_fast(lambdacursor, "lambda")
returns setof record
as '/home/clemens/masterarbeit/psql-autodiff/src/ext/autodiff_ext.so','autodiff_fast'
language C STRICT;


--test all functions and run them with their corresponding datatables
--explain analyze select * from label((select x, y from nums_label),(lambda(a)(pow(a.x, 2) * a.y + 5))) limit 10;
--select * from label_fast((select * from nums),(lambda(a)(atan2(a.x, a.y)))) limit 10;
--select * from kmeans((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
--select * from kmeans_threads((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
--select * from pagerank((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;
--select * from pagerank_threads((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;

select * from autodiff((select x, y, z from nums),(lambda(a)(a.x*a.x + 2 * a.y - a.z))) limit 10;
--select * from autodiff((select x, y from nums),(lambda(a)(-atan2(a.x, a.y)))) limit 10;
--select * from autodiff((select x, y from nums_null), (lambda(a)(a.x + a.y))) limit 10;
--select * from autodiff((select x, y, z from nums),lambda(a)((a.x*a.y) + a.z/2)); 

--select * from autodiff_fast((select x, y, z from nums_label), (lambda(a)(a.x + a.y * a.z))) limit 10;