--meta flags, jit needs to be on, load llvm loads some dependencies
set jit='on';
load 'llvmjit.so';

set jit_above_cost = 0;             --enforce jit-usage
set jit_inline_above_cost = 0;      --enforce jit-usage
set jit_optimize_above_cost = 0;    --enforce jit-usage

--create new tables and fill them with usable data
create table perftests(x float not null, y float not null, z float not null);
insert into perftests select generate_series(1, 10000000), generate_series(10000000, 1, -1), generate_series(1, 10000000);


--load all functions
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