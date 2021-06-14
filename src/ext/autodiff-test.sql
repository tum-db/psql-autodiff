--meta flags, jit needs to be on, load llvm loads some dependencies
set jit='on';
load 'llvmjit.so';


--drop all tables, to create new and fresh ones
drop table if exists nums;
drop table if exists nums_label;
drop table if exists points;
drop table if exists pages;


--create new tables and fill them with usable data
create table nums(x float not null, y float not null);
create table nums_label(x integer);
create table points(x float not null, y float not null);
create table pages(src float not null, dst float not null);

insert into nums select generate_series(1, 100), generate_series(101, 200);
insert into nums_label select generate_series(1,10);
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


--test all functions and run them with their corresponding datatables
--select * from label((select x from nums_label),(lambda(a)(a.x))) limit 10;
--select * from label_fast((select * from nums),(lambda(a)((a.x + a.y)/2))) limit 10;
--select * from kmeans((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
--select * from kmeans_threads((select * from points),(select * from points),(lambda(a,b)(a.x + a.y - (b.x + b.y))), 10, 100) limit 10;
--select * from pagerank((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;
--select * from pagerank_threads((select * from pages), (lambda(src)(src.src)), (lambda(dst)(dst.dst)), 0.85, 0.00001, 100, 100) limit 10;

select * from autodiff((select x from nums_label),(lambda(a)(a.x))) limit 10;