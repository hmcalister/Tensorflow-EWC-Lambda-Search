µљ
т¬
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ы
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8рх
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

: *
dtype0
А
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
: *
dtype0
И
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_35/kernel/v
Б
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:  *
dtype0
А
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
: *
dtype0
Й
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј( *'
shared_nameAdam/dense_34/kernel/v
В
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes
:	ј( *
dtype0
А
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/v
Й
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
Ь
"Adam/batch_normalization_33/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_33/beta/v
Х
6Adam/batch_normalization_33/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/v*
_output_shapes
:@*
dtype0
Ю
#Adam/batch_normalization_33/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_33/gamma/v
Ч
7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/v*
_output_shapes
:@*
dtype0
А
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/v
Й
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
Р
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/v
Й
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
Ь
"Adam/batch_normalization_32/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_32/beta/v
Х
6Adam/batch_normalization_32/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/v*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_32/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_32/gamma/v
Ч
7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/v*
_output_shapes
: *
dtype0
А
Adam/conv2d_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_0/bias/v
y
(Adam/conv2d_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_0/bias/v*
_output_shapes
: *
dtype0
Р
Adam/conv2d_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_0/kernel/v
Й
*Adam/conv2d_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_0/kernel/v*&
_output_shapes
: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

: *
dtype0
А
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
: *
dtype0
И
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_35/kernel/m
Б
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:  *
dtype0
А
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
: *
dtype0
Й
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј( *'
shared_nameAdam/dense_34/kernel/m
В
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes
:	ј( *
dtype0
А
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/m
Й
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
Ь
"Adam/batch_normalization_33/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_33/beta/m
Х
6Adam/batch_normalization_33/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/m*
_output_shapes
:@*
dtype0
Ю
#Adam/batch_normalization_33/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_33/gamma/m
Ч
7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/m*
_output_shapes
:@*
dtype0
А
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/m
Й
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
Р
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/m
Й
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
Ь
"Adam/batch_normalization_32/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_32/beta/m
Х
6Adam/batch_normalization_32/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/m*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_32/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_32/gamma/m
Ч
7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/m*
_output_shapes
: *
dtype0
А
Adam/conv2d_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_0/bias/m
y
(Adam/conv2d_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_0/bias/m*
_output_shapes
: *
dtype0
Р
Adam/conv2d_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_0/kernel/m
Й
*Adam/conv2d_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_0/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
: *
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:  *
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
: *
dtype0
{
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј( * 
shared_namedense_34/kernel
t
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes
:	ј( *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
В
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
§
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_33/moving_variance
Э
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
:@*
dtype0
Ь
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_33/moving_mean
Х
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
:@*
dtype0
О
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_33/beta
З
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
:@*
dtype0
Р
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_33/gamma
Й
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
:@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0
§
&batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_32/moving_variance
Э
:batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_32/moving_variance*
_output_shapes
: *
dtype0
Ь
"batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_32/moving_mean
Х
6batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_32/moving_mean*
_output_shapes
: *
dtype0
О
batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_32/beta
З
/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_32/beta*
_output_shapes
: *
dtype0
Р
batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_32/gamma
Й
0batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_32/gamma*
_output_shapes
: *
dtype0
r
conv2d_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_0/bias
k
!conv2d_0/bias/Read/ReadVariableOpReadVariableOpconv2d_0/bias*
_output_shapes
: *
dtype0
В
conv2d_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_0/kernel
{
#conv2d_0/kernel/Read/ReadVariableOpReadVariableOpconv2d_0/kernel*&
_output_shapes
: *
dtype0
Л
serving_default_input_17Placeholder*/
_output_shapes
:€€€€€€€€€  *
dtype0*$
shape:€€€€€€€€€  
ћ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17conv2d_0/kernelconv2d_0/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_3/kernelconv2d_3/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/biasdense/kernel
dense/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_99810837

NoOpNoOp
ЛЕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≈Д
valueЇДBґД BЃД
‘
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
'
#_self_saveable_object_factories* 
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories
  _jit_compiled_convolution_op*
≥
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
#'_self_saveable_object_factories* 
ъ
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.axis
	/gamma
0beta
1moving_mean
2moving_variance
#3_self_saveable_object_factories*
н
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
#<_self_saveable_object_factories
 =_jit_compiled_convolution_op*
н
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
#F_self_saveable_object_factories
 G_jit_compiled_convolution_op*
ъ
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
#S_self_saveable_object_factories*
н
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias
#\_self_saveable_object_factories
 ]_jit_compiled_convolution_op*
≥
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
#d_self_saveable_object_factories* 
Ћ
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
#m_self_saveable_object_factories*
Ћ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias
#v_self_saveable_object_factories*
¶
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias*
™
0
1
/2
03
14
25
:6
;7
D8
E9
O10
P11
Q12
R13
Z14
[15
k16
l17
t18
u19
}20
~21*
К
0
1
/2
03
:4
;5
D6
E7
O8
P9
Z10
[11
k12
l13
t14
u15
}16
~17*
* 
і
non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
Дtrace_0
Еtrace_1
Жtrace_2
Зtrace_3* 
:
Иtrace_0
Йtrace_1
Кtrace_2
Лtrace_3* 
* 
±
	Мiter
Нbeta_1
Оbeta_2

Пdecay
Рlearning_ratemоmп/mр0mс:mт;mуDmфEmхOmцPmчZmш[mщkmъlmыtmьumэ}mю~m€vАvБ/vВ0vГ:vД;vЕDvЖEvЗOvИPvЙZvК[vЛkvМlvНtvОuvП}vР~vС*

Сserving_default* 
* 

0
1*

0
1*
* 
Ш
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Чtrace_0* 

Шtrace_0* 
_Y
VARIABLE_VALUEconv2d_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
Ц
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

Юtrace_0* 

Яtrace_0* 
* 
 
/0
01
12
23*

/0
01*
* 
Ш
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

•trace_0
¶trace_1* 

Іtrace_0
®trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_32/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_32/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_32/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_32/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 

:0
;1*

:0
;1*
* 
Ш
©non_trainable_variables
™layers
Ђmetrics
 ђlayer_regularization_losses
≠layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

Ѓtrace_0* 

ѓtrace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

D0
E1*

D0
E1*
* 
Ш
∞non_trainable_variables
±layers
≤metrics
 ≥layer_regularization_losses
іlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

µtrace_0* 

ґtrace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
 
O0
P1
Q2
R3*

O0
P1*
* 
Ш
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

Љtrace_0
љtrace_1* 

Њtrace_0
њtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_33/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_33/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_33/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_33/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 

Z0
[1*

Z0
[1*
* 
Ш
јnon_trainable_variables
Ѕlayers
¬metrics
 √layer_regularization_losses
ƒlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

≈trace_0* 

∆trace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
Ц
«non_trainable_variables
»layers
…metrics
  layer_regularization_losses
Ћlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

ћtrace_0* 

Ќtrace_0* 
* 

k0
l1*

k0
l1*
* 
Ш
ќnon_trainable_variables
ѕlayers
–metrics
 —layer_regularization_losses
“layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

”trace_0* 

‘trace_0* 
_Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_34/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

t0
u1*

t0
u1*
* 
Ш
’non_trainable_variables
÷layers
„metrics
 Ўlayer_regularization_losses
ўlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

Џtrace_0* 

џtrace_0* 
_Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_35/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

}0
~1*

}0
~1*
* 
Ш
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
10
21
Q2
R3*
Z
0
1
2
3
4
5
6
7
	8

9
10
11*

г0
д1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

10
21*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
е	variables
ж	keras_api

зtotal

иcount*
M
й	variables
к	keras_api

лtotal

мcount
н
_fn_kwargs*

з0
и1*

е	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

л0
м1*

й	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
В|
VARIABLE_VALUEAdam/conv2d_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_32/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_33/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_32/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ПИ
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE"Adam/batch_normalization_33/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
И
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_0/kernel/Read/ReadVariableOp!conv2d_0/bias/Read/ReadVariableOp0batch_normalization_32/gamma/Read/ReadVariableOp/batch_normalization_32/beta/Read/ReadVariableOp6batch_normalization_32/moving_mean/Read/ReadVariableOp:batch_normalization_32/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_0/kernel/m/Read/ReadVariableOp(Adam/conv2d_0/bias/m/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_32/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_33/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp*Adam/conv2d_0/kernel/v/Read/ReadVariableOp(Adam/conv2d_0/bias/v/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_32/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_33/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_save_99811609
«
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_0/kernelconv2d_0/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_3/kernelconv2d_3/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_0/kernel/mAdam/conv2d_0/bias/m#Adam/batch_normalization_32/gamma/m"Adam/batch_normalization_32/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/m#Adam/batch_normalization_33/gamma/m"Adam/batch_normalization_33/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d_0/kernel/vAdam/conv2d_0/bias/v#Adam/batch_normalization_32/gamma/v"Adam/batch_normalization_32/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v#Adam/batch_normalization_33/gamma/v"Adam/batch_normalization_33/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/vAdam/dense/kernel/vAdam/dense/bias/v*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference__traced_restore_99811820ВЄ
Ж
€
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99810242

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ж
€
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99811233

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
…
Ш
+__inference_dense_35_layer_call_fn_99811355

inputs
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_99810310o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Хs
д
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811101

inputsA
'conv2d_0_conv2d_readvariableop_resource: 6
(conv2d_0_biasadd_readvariableop_resource: <
.batch_normalization_32_readvariableop_resource: >
0batch_normalization_32_readvariableop_1_resource: M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@<
.batch_normalization_33_readvariableop_resource:@>
0batch_normalization_33_readvariableop_1_resource:@M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@:
'dense_34_matmul_readvariableop_resource:	ј( 6
(dense_34_biasadd_readvariableop_resource: 9
'dense_35_matmul_readvariableop_resource:  6
(dense_35_biasadd_readvariableop_resource: 6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource:
identityИҐ%batch_normalization_32/AssignNewValueҐ'batch_normalization_32/AssignNewValue_1Ґ6batch_normalization_32/FusedBatchNormV3/ReadVariableOpҐ8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Ґ%batch_normalization_32/ReadVariableOpҐ'batch_normalization_32/ReadVariableOp_1Ґ%batch_normalization_33/AssignNewValueҐ'batch_normalization_33/AssignNewValue_1Ґ6batch_normalization_33/FusedBatchNormV3/ReadVariableOpҐ8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Ґ%batch_normalization_33/ReadVariableOpҐ'batch_normalization_33/ReadVariableOp_1Ґconv2d_0/BiasAdd/ReadVariableOpҐconv2d_0/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_34/BiasAdd/ReadVariableOpҐdense_34/MatMul/ReadVariableOpҐdense_35/BiasAdd/ReadVariableOpҐdense_35/MatMul/ReadVariableOpО
conv2d_0/Conv2D/ReadVariableOpReadVariableOp'conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ђ
conv2d_0/Conv2DConv2Dinputs&conv2d_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Д
conv2d_0/BiasAdd/ReadVariableOpReadVariableOp(conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_0/BiasAddBiasAddconv2d_0/Conv2D:output:0'conv2d_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ j
conv2d_0/ReluReluconv2d_0/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ≠
max_pooling2d_16/MaxPoolMaxPoolconv2d_0/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
Р
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
: *
dtype0≤
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ґ
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0“
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_16/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<Ґ
%batch_normalization_32/AssignNewValueAssignVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource4batch_normalization_32/FusedBatchNormV3:batch_mean:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ђ
'batch_normalization_32/AssignNewValue_1AssignVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_32/FusedBatchNormV3:batch_variance:09^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0—
conv2d_1/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ѕ
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Р
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:@*
dtype0≤
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ґ
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<Ґ
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ђ
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(О
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0—
conv2d_3/Conv2DConv2D+batch_normalization_33/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@*
paddingVALID*
strides
Д
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€		@a
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  И
flatten_16/ReshapeReshapeconv2d_3/Relu:activations:0flatten_16/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(З
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes
:	ј( *
dtype0Р
dense_34/MatMulMatMulflatten_16/Reshape:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Р
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_35/ReluReludense_35/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0К
dense/MatMulMatMuldense_35/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€≈
NoOpNoOp&^batch_normalization_32/AssignNewValue(^batch_normalization_32/AssignNewValue_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1 ^conv2d_0/BiasAdd/ReadVariableOp^conv2d_0/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_32/AssignNewValue%batch_normalization_32/AssignNewValue2R
'batch_normalization_32/AssignNewValue_1'batch_normalization_32/AssignNewValue_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12B
conv2d_0/BiasAdd/ReadVariableOpconv2d_0/BiasAdd/ReadVariableOp2@
conv2d_0/Conv2D/ReadVariableOpconv2d_0/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
–
‘
/__inference_task_1_model_layer_call_fn_99810886

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:	ј( 

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
с
†
+__inference_conv2d_3_layer_call_fn_99811304

inputs!
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99810268w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€		@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ц
j
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99810049

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ж
€
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99810198

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
 
d
H__inference_flatten_16_layer_call_and_return_conditional_losses_99811326

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€		@:W S
/
_output_shapes
:€€€€€€€€€		@
 
_user_specified_nameinputs
°

ш
F__inference_dense_34_layer_call_and_return_conditional_losses_99810293

inputs1
matmul_readvariableop_resource:	ј( -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј( *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј(
 
_user_specified_nameinputs
Ъ	
‘
9__inference_batch_normalization_33_layer_call_fn_99811246

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810138Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Й
√
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810105

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
Я
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810074

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
с
†
+__inference_conv2d_0_layer_call_fn_99811110

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99810198w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
Ж
€
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99811121

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
К;
Љ

J__inference_task_1_model_layer_call_and_return_conditional_losses_99810721
input_17+
conv2d_0_99810665: 
conv2d_0_99810667: -
batch_normalization_32_99810671: -
batch_normalization_32_99810673: -
batch_normalization_32_99810675: -
batch_normalization_32_99810677: +
conv2d_1_99810680:  
conv2d_1_99810682: +
conv2d_2_99810685: @
conv2d_2_99810687:@-
batch_normalization_33_99810690:@-
batch_normalization_33_99810692:@-
batch_normalization_33_99810694:@-
batch_normalization_33_99810696:@+
conv2d_3_99810699:@@
conv2d_3_99810701:@$
dense_34_99810705:	ј( 
dense_34_99810707: #
dense_35_99810710:  
dense_35_99810712:  
dense_99810715: 
dense_99810717:
identityИҐ.batch_normalization_32/StatefulPartitionedCallҐ.batch_normalization_33/StatefulPartitionedCallҐ conv2d_0/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐ dense_34/StatefulPartitionedCallҐ dense_35/StatefulPartitionedCallГ
 conv2d_0/StatefulPartitionedCallStatefulPartitionedCallinput_17conv2d_0_99810665conv2d_0_99810667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99810198ш
 max_pooling2d_16/PartitionedCallPartitionedCall)conv2d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99810049Ґ
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0batch_normalization_32_99810671batch_normalization_32_99810673batch_normalization_32_99810675batch_normalization_32_99810677*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810074≤
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_1_99810680conv2d_1_99810682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99810225§
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_99810685conv2d_2_99810687*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99810242Ґ
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_33_99810690batch_normalization_33_99810692batch_normalization_33_99810694batch_normalization_33_99810696*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810138≤
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_3_99810699conv2d_3_99810701*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99810268е
flatten_16/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_99810280Ц
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_34_99810705dense_34_99810707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_99810293Ь
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_99810710dense_35_99810712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_99810310Р
dense/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0dense_99810715dense_99810717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_99810326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall!^conv2d_0/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2D
 conv2d_0/StatefulPartitionedCall conv2d_0/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_17
ћ
Щ
+__inference_dense_34_layer_call_fn_99811335

inputs
unknown:	ј( 
	unknown_0: 
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_99810293o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј(
 
_user_specified_nameinputs
÷
÷
/__inference_task_1_model_layer_call_fn_99810380
input_17!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:	ј( 

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_17
ѕ
Я
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811175

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
°

ш
F__inference_dense_34_layer_call_and_return_conditional_losses_99811346

inputs1
matmul_readvariableop_resource:	ј( -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј( *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј(
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_layer_call_and_return_conditional_losses_99810326

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
 
d
H__inference_flatten_16_layer_call_and_return_conditional_losses_99810280

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€		@:W S
/
_output_shapes
:€€€€€€€€€		@
 
_user_specified_nameinputs
∆	
ф
C__inference_dense_layer_call_and_return_conditional_losses_99811385

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
¶
Ќ
&__inference_signature_wrapper_99810837
input_17!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:	ј( 

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_99810040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_17
“
÷
/__inference_task_1_model_layer_call_fn_99810662
input_17!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:	ј( 

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810566o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_17
Ж
€
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99811213

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
Я
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810138

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
£{
„
#__inference__wrapped_model_99810040
input_17N
4task_1_model_conv2d_0_conv2d_readvariableop_resource: C
5task_1_model_conv2d_0_biasadd_readvariableop_resource: I
;task_1_model_batch_normalization_32_readvariableop_resource: K
=task_1_model_batch_normalization_32_readvariableop_1_resource: Z
Ltask_1_model_batch_normalization_32_fusedbatchnormv3_readvariableop_resource: \
Ntask_1_model_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource: N
4task_1_model_conv2d_1_conv2d_readvariableop_resource:  C
5task_1_model_conv2d_1_biasadd_readvariableop_resource: N
4task_1_model_conv2d_2_conv2d_readvariableop_resource: @C
5task_1_model_conv2d_2_biasadd_readvariableop_resource:@I
;task_1_model_batch_normalization_33_readvariableop_resource:@K
=task_1_model_batch_normalization_33_readvariableop_1_resource:@Z
Ltask_1_model_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:@\
Ntask_1_model_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:@N
4task_1_model_conv2d_3_conv2d_readvariableop_resource:@@C
5task_1_model_conv2d_3_biasadd_readvariableop_resource:@G
4task_1_model_dense_34_matmul_readvariableop_resource:	ј( C
5task_1_model_dense_34_biasadd_readvariableop_resource: F
4task_1_model_dense_35_matmul_readvariableop_resource:  C
5task_1_model_dense_35_biasadd_readvariableop_resource: C
1task_1_model_dense_matmul_readvariableop_resource: @
2task_1_model_dense_biasadd_readvariableop_resource:
identityИҐCtask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOpҐEtask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Ґ2task_1_model/batch_normalization_32/ReadVariableOpҐ4task_1_model/batch_normalization_32/ReadVariableOp_1ҐCtask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOpҐEtask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Ґ2task_1_model/batch_normalization_33/ReadVariableOpҐ4task_1_model/batch_normalization_33/ReadVariableOp_1Ґ,task_1_model/conv2d_0/BiasAdd/ReadVariableOpҐ+task_1_model/conv2d_0/Conv2D/ReadVariableOpҐ,task_1_model/conv2d_1/BiasAdd/ReadVariableOpҐ+task_1_model/conv2d_1/Conv2D/ReadVariableOpҐ,task_1_model/conv2d_2/BiasAdd/ReadVariableOpҐ+task_1_model/conv2d_2/Conv2D/ReadVariableOpҐ,task_1_model/conv2d_3/BiasAdd/ReadVariableOpҐ+task_1_model/conv2d_3/Conv2D/ReadVariableOpҐ)task_1_model/dense/BiasAdd/ReadVariableOpҐ(task_1_model/dense/MatMul/ReadVariableOpҐ,task_1_model/dense_34/BiasAdd/ReadVariableOpҐ+task_1_model/dense_34/MatMul/ReadVariableOpҐ,task_1_model/dense_35/BiasAdd/ReadVariableOpҐ+task_1_model/dense_35/MatMul/ReadVariableOp®
+task_1_model/conv2d_0/Conv2D/ReadVariableOpReadVariableOp4task_1_model_conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0»
task_1_model/conv2d_0/Conv2DConv2Dinput_173task_1_model/conv2d_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Ю
,task_1_model/conv2d_0/BiasAdd/ReadVariableOpReadVariableOp5task_1_model_conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0њ
task_1_model/conv2d_0/BiasAddBiasAdd%task_1_model/conv2d_0/Conv2D:output:04task_1_model/conv2d_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ Д
task_1_model/conv2d_0/ReluRelu&task_1_model/conv2d_0/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ «
%task_1_model/max_pooling2d_16/MaxPoolMaxPool(task_1_model/conv2d_0/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
™
2task_1_model/batch_normalization_32/ReadVariableOpReadVariableOp;task_1_model_batch_normalization_32_readvariableop_resource*
_output_shapes
: *
dtype0Ѓ
4task_1_model/batch_normalization_32/ReadVariableOp_1ReadVariableOp=task_1_model_batch_normalization_32_readvariableop_1_resource*
_output_shapes
: *
dtype0ћ
Ctask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpLtask_1_model_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0–
Etask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNtask_1_model_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Т
4task_1_model/batch_normalization_32/FusedBatchNormV3FusedBatchNormV3.task_1_model/max_pooling2d_16/MaxPool:output:0:task_1_model/batch_normalization_32/ReadVariableOp:value:0<task_1_model/batch_normalization_32/ReadVariableOp_1:value:0Ktask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Mtask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( ®
+task_1_model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4task_1_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ш
task_1_model/conv2d_1/Conv2DConv2D8task_1_model/batch_normalization_32/FusedBatchNormV3:y:03task_1_model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Ю
,task_1_model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5task_1_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0њ
task_1_model/conv2d_1/BiasAddBiasAdd%task_1_model/conv2d_1/Conv2D:output:04task_1_model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ Д
task_1_model/conv2d_1/ReluRelu&task_1_model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ®
+task_1_model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4task_1_model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0и
task_1_model/conv2d_2/Conv2DConv2D(task_1_model/conv2d_1/Relu:activations:03task_1_model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
Ю
,task_1_model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5task_1_model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
task_1_model/conv2d_2/BiasAddBiasAdd%task_1_model/conv2d_2/Conv2D:output:04task_1_model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@Д
task_1_model/conv2d_2/ReluRelu&task_1_model/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@™
2task_1_model/batch_normalization_33/ReadVariableOpReadVariableOp;task_1_model_batch_normalization_33_readvariableop_resource*
_output_shapes
:@*
dtype0Ѓ
4task_1_model/batch_normalization_33/ReadVariableOp_1ReadVariableOp=task_1_model_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:@*
dtype0ћ
Ctask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpLtask_1_model_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0–
Etask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNtask_1_model_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0М
4task_1_model/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3(task_1_model/conv2d_2/Relu:activations:0:task_1_model/batch_normalization_33/ReadVariableOp:value:0<task_1_model/batch_normalization_33/ReadVariableOp_1:value:0Ktask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Mtask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( ®
+task_1_model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4task_1_model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ш
task_1_model/conv2d_3/Conv2DConv2D8task_1_model/batch_normalization_33/FusedBatchNormV3:y:03task_1_model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@*
paddingVALID*
strides
Ю
,task_1_model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5task_1_model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0њ
task_1_model/conv2d_3/BiasAddBiasAdd%task_1_model/conv2d_3/Conv2D:output:04task_1_model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@Д
task_1_model/conv2d_3/ReluRelu&task_1_model/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€		@n
task_1_model/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  ѓ
task_1_model/flatten_16/ReshapeReshape(task_1_model/conv2d_3/Relu:activations:0&task_1_model/flatten_16/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(°
+task_1_model/dense_34/MatMul/ReadVariableOpReadVariableOp4task_1_model_dense_34_matmul_readvariableop_resource*
_output_shapes
:	ј( *
dtype0Ј
task_1_model/dense_34/MatMulMatMul(task_1_model/flatten_16/Reshape:output:03task_1_model/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
,task_1_model/dense_34/BiasAdd/ReadVariableOpReadVariableOp5task_1_model_dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Є
task_1_model/dense_34/BiasAddBiasAdd&task_1_model/dense_34/MatMul:product:04task_1_model/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ |
task_1_model/dense_34/ReluRelu&task_1_model/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ †
+task_1_model/dense_35/MatMul/ReadVariableOpReadVariableOp4task_1_model_dense_35_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Ј
task_1_model/dense_35/MatMulMatMul(task_1_model/dense_34/Relu:activations:03task_1_model/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ю
,task_1_model/dense_35/BiasAdd/ReadVariableOpReadVariableOp5task_1_model_dense_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Є
task_1_model/dense_35/BiasAddBiasAdd&task_1_model/dense_35/MatMul:product:04task_1_model/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ |
task_1_model/dense_35/ReluRelu&task_1_model/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
(task_1_model/dense/MatMul/ReadVariableOpReadVariableOp1task_1_model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0±
task_1_model/dense/MatMulMatMul(task_1_model/dense_35/Relu:activations:00task_1_model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)task_1_model/dense/BiasAdd/ReadVariableOpReadVariableOp2task_1_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
task_1_model/dense/BiasAddBiasAdd#task_1_model/dense/MatMul:product:01task_1_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
IdentityIdentity#task_1_model/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€њ	
NoOpNoOpD^task_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOpF^task_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_13^task_1_model/batch_normalization_32/ReadVariableOp5^task_1_model/batch_normalization_32/ReadVariableOp_1D^task_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOpF^task_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_13^task_1_model/batch_normalization_33/ReadVariableOp5^task_1_model/batch_normalization_33/ReadVariableOp_1-^task_1_model/conv2d_0/BiasAdd/ReadVariableOp,^task_1_model/conv2d_0/Conv2D/ReadVariableOp-^task_1_model/conv2d_1/BiasAdd/ReadVariableOp,^task_1_model/conv2d_1/Conv2D/ReadVariableOp-^task_1_model/conv2d_2/BiasAdd/ReadVariableOp,^task_1_model/conv2d_2/Conv2D/ReadVariableOp-^task_1_model/conv2d_3/BiasAdd/ReadVariableOp,^task_1_model/conv2d_3/Conv2D/ReadVariableOp*^task_1_model/dense/BiasAdd/ReadVariableOp)^task_1_model/dense/MatMul/ReadVariableOp-^task_1_model/dense_34/BiasAdd/ReadVariableOp,^task_1_model/dense_34/MatMul/ReadVariableOp-^task_1_model/dense_35/BiasAdd/ReadVariableOp,^task_1_model/dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 2К
Ctask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOpCtask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2О
Etask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Etask_1_model/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12h
2task_1_model/batch_normalization_32/ReadVariableOp2task_1_model/batch_normalization_32/ReadVariableOp2l
4task_1_model/batch_normalization_32/ReadVariableOp_14task_1_model/batch_normalization_32/ReadVariableOp_12К
Ctask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOpCtask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2О
Etask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Etask_1_model/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12h
2task_1_model/batch_normalization_33/ReadVariableOp2task_1_model/batch_normalization_33/ReadVariableOp2l
4task_1_model/batch_normalization_33/ReadVariableOp_14task_1_model/batch_normalization_33/ReadVariableOp_12\
,task_1_model/conv2d_0/BiasAdd/ReadVariableOp,task_1_model/conv2d_0/BiasAdd/ReadVariableOp2Z
+task_1_model/conv2d_0/Conv2D/ReadVariableOp+task_1_model/conv2d_0/Conv2D/ReadVariableOp2\
,task_1_model/conv2d_1/BiasAdd/ReadVariableOp,task_1_model/conv2d_1/BiasAdd/ReadVariableOp2Z
+task_1_model/conv2d_1/Conv2D/ReadVariableOp+task_1_model/conv2d_1/Conv2D/ReadVariableOp2\
,task_1_model/conv2d_2/BiasAdd/ReadVariableOp,task_1_model/conv2d_2/BiasAdd/ReadVariableOp2Z
+task_1_model/conv2d_2/Conv2D/ReadVariableOp+task_1_model/conv2d_2/Conv2D/ReadVariableOp2\
,task_1_model/conv2d_3/BiasAdd/ReadVariableOp,task_1_model/conv2d_3/BiasAdd/ReadVariableOp2Z
+task_1_model/conv2d_3/Conv2D/ReadVariableOp+task_1_model/conv2d_3/Conv2D/ReadVariableOp2V
)task_1_model/dense/BiasAdd/ReadVariableOp)task_1_model/dense/BiasAdd/ReadVariableOp2T
(task_1_model/dense/MatMul/ReadVariableOp(task_1_model/dense/MatMul/ReadVariableOp2\
,task_1_model/dense_34/BiasAdd/ReadVariableOp,task_1_model/dense_34/BiasAdd/ReadVariableOp2Z
+task_1_model/dense_34/MatMul/ReadVariableOp+task_1_model/dense_34/MatMul/ReadVariableOp2\
,task_1_model/dense_35/BiasAdd/ReadVariableOp,task_1_model/dense_35/BiasAdd/ReadVariableOp2Z
+task_1_model/dense_35/MatMul/ReadVariableOp+task_1_model/dense_35/MatMul/ReadVariableOp:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_17
Ж
€
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99811315

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€		@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€		@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Э

ч
F__inference_dense_35_layer_call_and_return_conditional_losses_99810310

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ш	
‘
9__inference_batch_normalization_33_layer_call_fn_99811259

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810169Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Й
√
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811295

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Э

ч
F__inference_dense_35_layer_call_and_return_conditional_losses_99811366

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
А;
Ї

J__inference_task_1_model_layer_call_and_return_conditional_losses_99810566

inputs+
conv2d_0_99810510: 
conv2d_0_99810512: -
batch_normalization_32_99810516: -
batch_normalization_32_99810518: -
batch_normalization_32_99810520: -
batch_normalization_32_99810522: +
conv2d_1_99810525:  
conv2d_1_99810527: +
conv2d_2_99810530: @
conv2d_2_99810532:@-
batch_normalization_33_99810535:@-
batch_normalization_33_99810537:@-
batch_normalization_33_99810539:@-
batch_normalization_33_99810541:@+
conv2d_3_99810544:@@
conv2d_3_99810546:@$
dense_34_99810550:	ј( 
dense_34_99810552: #
dense_35_99810555:  
dense_35_99810557:  
dense_99810560: 
dense_99810562:
identityИҐ.batch_normalization_32/StatefulPartitionedCallҐ.batch_normalization_33/StatefulPartitionedCallҐ conv2d_0/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐ dense_34/StatefulPartitionedCallҐ dense_35/StatefulPartitionedCallБ
 conv2d_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_0_99810510conv2d_0_99810512*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99810198ш
 max_pooling2d_16/PartitionedCallPartitionedCall)conv2d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99810049†
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0batch_normalization_32_99810516batch_normalization_32_99810518batch_normalization_32_99810520batch_normalization_32_99810522*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810105≤
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_1_99810525conv2d_1_99810527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99810225§
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_99810530conv2d_2_99810532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99810242†
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_33_99810535batch_normalization_33_99810537batch_normalization_33_99810539batch_normalization_33_99810541*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810169≤
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_3_99810544conv2d_3_99810546*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99810268е
flatten_16/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_99810280Ц
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_34_99810550dense_34_99810552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_99810293Ь
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_99810555dense_35_99810557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_99810310Р
dense/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0dense_99810560dense_99810562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_99810326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall!^conv2d_0/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2D
 conv2d_0/StatefulPartitionedCall conv2d_0/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
ЎМ
„*
$__inference__traced_restore_99811820
file_prefix:
 assignvariableop_conv2d_0_kernel: .
 assignvariableop_1_conv2d_0_bias: =
/assignvariableop_2_batch_normalization_32_gamma: <
.assignvariableop_3_batch_normalization_32_beta: C
5assignvariableop_4_batch_normalization_32_moving_mean: G
9assignvariableop_5_batch_normalization_32_moving_variance: <
"assignvariableop_6_conv2d_1_kernel:  .
 assignvariableop_7_conv2d_1_bias: <
"assignvariableop_8_conv2d_2_kernel: @.
 assignvariableop_9_conv2d_2_bias:@>
0assignvariableop_10_batch_normalization_33_gamma:@=
/assignvariableop_11_batch_normalization_33_beta:@D
6assignvariableop_12_batch_normalization_33_moving_mean:@H
:assignvariableop_13_batch_normalization_33_moving_variance:@=
#assignvariableop_14_conv2d_3_kernel:@@/
!assignvariableop_15_conv2d_3_bias:@6
#assignvariableop_16_dense_34_kernel:	ј( /
!assignvariableop_17_dense_34_bias: 5
#assignvariableop_18_dense_35_kernel:  /
!assignvariableop_19_dense_35_bias: 2
 assignvariableop_20_dense_kernel: ,
assignvariableop_21_dense_bias:'
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: #
assignvariableop_29_total: #
assignvariableop_30_count: D
*assignvariableop_31_adam_conv2d_0_kernel_m: 6
(assignvariableop_32_adam_conv2d_0_bias_m: E
7assignvariableop_33_adam_batch_normalization_32_gamma_m: D
6assignvariableop_34_adam_batch_normalization_32_beta_m: D
*assignvariableop_35_adam_conv2d_1_kernel_m:  6
(assignvariableop_36_adam_conv2d_1_bias_m: D
*assignvariableop_37_adam_conv2d_2_kernel_m: @6
(assignvariableop_38_adam_conv2d_2_bias_m:@E
7assignvariableop_39_adam_batch_normalization_33_gamma_m:@D
6assignvariableop_40_adam_batch_normalization_33_beta_m:@D
*assignvariableop_41_adam_conv2d_3_kernel_m:@@6
(assignvariableop_42_adam_conv2d_3_bias_m:@=
*assignvariableop_43_adam_dense_34_kernel_m:	ј( 6
(assignvariableop_44_adam_dense_34_bias_m: <
*assignvariableop_45_adam_dense_35_kernel_m:  6
(assignvariableop_46_adam_dense_35_bias_m: 9
'assignvariableop_47_adam_dense_kernel_m: 3
%assignvariableop_48_adam_dense_bias_m:D
*assignvariableop_49_adam_conv2d_0_kernel_v: 6
(assignvariableop_50_adam_conv2d_0_bias_v: E
7assignvariableop_51_adam_batch_normalization_32_gamma_v: D
6assignvariableop_52_adam_batch_normalization_32_beta_v: D
*assignvariableop_53_adam_conv2d_1_kernel_v:  6
(assignvariableop_54_adam_conv2d_1_bias_v: D
*assignvariableop_55_adam_conv2d_2_kernel_v: @6
(assignvariableop_56_adam_conv2d_2_bias_v:@E
7assignvariableop_57_adam_batch_normalization_33_gamma_v:@D
6assignvariableop_58_adam_batch_normalization_33_beta_v:@D
*assignvariableop_59_adam_conv2d_3_kernel_v:@@6
(assignvariableop_60_adam_conv2d_3_bias_v:@=
*assignvariableop_61_adam_dense_34_kernel_v:	ј( 6
(assignvariableop_62_adam_dense_34_bias_v: <
*assignvariableop_63_adam_dense_35_kernel_v:  6
(assignvariableop_64_adam_dense_35_bias_v: 9
'assignvariableop_65_adam_dense_kernel_v: 3
%assignvariableop_66_adam_dense_bias_v:
identity_68ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9 %
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*р$
valueж$Bг$DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHы
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*Э
valueУBРDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B х
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapesУ
Р::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOpAssignVariableOp assignvariableop_conv2d_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_32_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_32_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_32_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_32_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_33_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_33_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_33_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ђ
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_33_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_34_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_34_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_35_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_35_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_21AssignVariableOpassignvariableop_21_dense_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_0_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_0_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_batch_normalization_32_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_34AssignVariableOp6assignvariableop_34_adam_batch_normalization_32_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_batch_normalization_33_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_batch_normalization_33_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_34_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_34_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_35_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_35_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adam_dense_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_48AssignVariableOp%assignvariableop_48_adam_dense_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_0_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_0_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_batch_normalization_32_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_32_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv2d_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv2d_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_33_gamma_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_33_beta_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_3_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_3_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_34_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_34_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_35_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_35_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_dense_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_66AssignVariableOp%assignvariableop_66_adam_dense_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 С
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_68IdentityIdentity_67:output:0^NoOp_1*
T0*
_output_shapes
: ю
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_68Identity_68:output:0*Э
_input_shapesЛ
И: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Й
√
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811193

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ц
j
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99811131

inputs
identityҐ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ш	
‘
9__inference_batch_normalization_32_layer_call_fn_99811157

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810105Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ѕ
O
3__inference_max_pooling2d_16_layer_call_fn_99811126

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99810049Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ћ
‘
/__inference_task_1_model_layer_call_fn_99810935

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@@

unknown_14:@

unknown_15:	ј( 

unknown_16: 

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810566o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
√
Х
(__inference_dense_layer_call_fn_99811375

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_99810326o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
Я
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811277

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ж
€
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99810225

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ї
I
-__inference_flatten_16_layer_call_fn_99811320

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_99810280a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€		@:W S
/
_output_shapes
:€€€€€€€€€		@
 
_user_specified_nameinputs
Ж
€
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99810268

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€		@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€		@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ѕd
ј
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811018

inputsA
'conv2d_0_conv2d_readvariableop_resource: 6
(conv2d_0_biasadd_readvariableop_resource: <
.batch_normalization_32_readvariableop_resource: >
0batch_normalization_32_readvariableop_1_resource: M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@<
.batch_normalization_33_readvariableop_resource:@>
0batch_normalization_33_readvariableop_1_resource:@M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@:
'dense_34_matmul_readvariableop_resource:	ј( 6
(dense_34_biasadd_readvariableop_resource: 9
'dense_35_matmul_readvariableop_resource:  6
(dense_35_biasadd_readvariableop_resource: 6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource:
identityИҐ6batch_normalization_32/FusedBatchNormV3/ReadVariableOpҐ8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Ґ%batch_normalization_32/ReadVariableOpҐ'batch_normalization_32/ReadVariableOp_1Ґ6batch_normalization_33/FusedBatchNormV3/ReadVariableOpҐ8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Ґ%batch_normalization_33/ReadVariableOpҐ'batch_normalization_33/ReadVariableOp_1Ґconv2d_0/BiasAdd/ReadVariableOpҐconv2d_0/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_34/BiasAdd/ReadVariableOpҐdense_34/MatMul/ReadVariableOpҐdense_35/BiasAdd/ReadVariableOpҐdense_35/MatMul/ReadVariableOpО
conv2d_0/Conv2D/ReadVariableOpReadVariableOp'conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ђ
conv2d_0/Conv2DConv2Dinputs&conv2d_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Д
conv2d_0/BiasAdd/ReadVariableOpReadVariableOp(conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_0/BiasAddBiasAddconv2d_0/Conv2D:output:0'conv2d_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ j
conv2d_0/ReluReluconv2d_0/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ ≠
max_pooling2d_16/MaxPoolMaxPoolconv2d_0/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
Р
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
: *
dtype0Ф
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
: *
dtype0≤
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ґ
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ƒ
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_16/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0—
conv2d_1/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ѕ
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@Р
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:@*
dtype0≤
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ґ
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( О
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0—
conv2d_3/Conv2DConv2D+batch_normalization_33/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@*
paddingVALID*
strides
Д
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€		@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€		@a
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  И
flatten_16/ReshapeReshapeconv2d_3/Relu:activations:0flatten_16/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(З
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes
:	ј( *
dtype0Р
dense_34/MatMulMatMulflatten_16/Reshape:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ж
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0Р
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Д
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0С
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ b
dense_35/ReluReludense_35/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype0К
dense/MatMulMatMuldense_35/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€°
NoOpNoOp7^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1 ^conv2d_0/BiasAdd/ReadVariableOp^conv2d_0/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12B
conv2d_0/BiasAdd/ReadVariableOpconv2d_0/BiasAdd/ReadVariableOp2@
conv2d_0/Conv2D/ReadVariableOpconv2d_0/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
Й
√
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810169

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
с
†
+__inference_conv2d_2_layer_call_fn_99811222

inputs!
unknown: @
	unknown_0:@
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99810242w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ъ	
‘
9__inference_batch_normalization_32_layer_call_fn_99811144

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810074Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
њД
ф
!__inference__traced_save_99811609
file_prefix.
*savev2_conv2d_0_kernel_read_readvariableop,
(savev2_conv2d_0_bias_read_readvariableop;
7savev2_batch_normalization_32_gamma_read_readvariableop:
6savev2_batch_normalization_32_beta_read_readvariableopA
=savev2_batch_normalization_32_moving_mean_read_readvariableopE
Asavev2_batch_normalization_32_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_0_kernel_m_read_readvariableop3
/savev2_adam_conv2d_0_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop5
1savev2_adam_conv2d_0_kernel_v_read_readvariableop3
/savev2_adam_conv2d_0_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: «%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*р$
valueж$Bг$DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHш
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*Э
valueУBРDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B н
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_0_kernel_read_readvariableop(savev2_conv2d_0_bias_read_readvariableop7savev2_batch_normalization_32_gamma_read_readvariableop6savev2_batch_normalization_32_beta_read_readvariableop=savev2_batch_normalization_32_moving_mean_read_readvariableopAsavev2_batch_normalization_32_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_0_kernel_m_read_readvariableop/savev2_adam_conv2d_0_bias_m_read_readvariableop>savev2_adam_batch_normalization_32_gamma_m_read_readvariableop=savev2_adam_batch_normalization_32_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop>savev2_adam_batch_normalization_33_gamma_m_read_readvariableop=savev2_adam_batch_normalization_33_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop1savev2_adam_conv2d_0_kernel_v_read_readvariableop/savev2_adam_conv2d_0_bias_v_read_readvariableop>savev2_adam_batch_normalization_32_gamma_v_read_readvariableop=savev2_adam_batch_normalization_32_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop>savev2_adam_batch_normalization_33_gamma_v_read_readvariableop=savev2_adam_batch_normalization_33_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Њ
_input_shapesђ
©: : : : : : : :  : : @:@:@:@:@:@:@@:@:	ј( : :  : : :: : : : : : : : : : : : : :  : : @:@:@:@:@@:@:	ј( : :  : : :: : : : :  : : @:@:@:@:@@:@:	ј( : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	ј( : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:,*(
&
_output_shapes
:@@: +

_output_shapes
:@:%,!

_output_shapes
:	ј( : -

_output_shapes
: :$. 

_output_shapes

:  : /

_output_shapes
: :$0 

_output_shapes

: : 1

_output_shapes
::,2(
&
_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: :,6(
&
_output_shapes
:  : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:,<(
&
_output_shapes
:@@: =

_output_shapes
:@:%>!

_output_shapes
:	ј( : ?

_output_shapes
: :$@ 

_output_shapes

:  : A

_output_shapes
: :$B 

_output_shapes

: : C

_output_shapes
::D

_output_shapes
: 
Д;
Ї

J__inference_task_1_model_layer_call_and_return_conditional_losses_99810333

inputs+
conv2d_0_99810199: 
conv2d_0_99810201: -
batch_normalization_32_99810205: -
batch_normalization_32_99810207: -
batch_normalization_32_99810209: -
batch_normalization_32_99810211: +
conv2d_1_99810226:  
conv2d_1_99810228: +
conv2d_2_99810243: @
conv2d_2_99810245:@-
batch_normalization_33_99810248:@-
batch_normalization_33_99810250:@-
batch_normalization_33_99810252:@-
batch_normalization_33_99810254:@+
conv2d_3_99810269:@@
conv2d_3_99810271:@$
dense_34_99810294:	ј( 
dense_34_99810296: #
dense_35_99810311:  
dense_35_99810313:  
dense_99810327: 
dense_99810329:
identityИҐ.batch_normalization_32/StatefulPartitionedCallҐ.batch_normalization_33/StatefulPartitionedCallҐ conv2d_0/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐ dense_34/StatefulPartitionedCallҐ dense_35/StatefulPartitionedCallБ
 conv2d_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_0_99810199conv2d_0_99810201*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99810198ш
 max_pooling2d_16/PartitionedCallPartitionedCall)conv2d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99810049Ґ
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0batch_normalization_32_99810205batch_normalization_32_99810207batch_normalization_32_99810209batch_normalization_32_99810211*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810074≤
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_1_99810226conv2d_1_99810228*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99810225§
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_99810243conv2d_2_99810245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99810242Ґ
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_33_99810248batch_normalization_33_99810250batch_normalization_33_99810252batch_normalization_33_99810254*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810138≤
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_3_99810269conv2d_3_99810271*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99810268е
flatten_16/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_99810280Ц
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_34_99810294dense_34_99810296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_99810293Ь
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_99810311dense_35_99810313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_99810310Р
dense/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0dense_99810327dense_99810329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_99810326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall!^conv2d_0/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2D
 conv2d_0/StatefulPartitionedCall conv2d_0/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs
с
†
+__inference_conv2d_1_layer_call_fn_99811202

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99810225w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ж;
Љ

J__inference_task_1_model_layer_call_and_return_conditional_losses_99810780
input_17+
conv2d_0_99810724: 
conv2d_0_99810726: -
batch_normalization_32_99810730: -
batch_normalization_32_99810732: -
batch_normalization_32_99810734: -
batch_normalization_32_99810736: +
conv2d_1_99810739:  
conv2d_1_99810741: +
conv2d_2_99810744: @
conv2d_2_99810746:@-
batch_normalization_33_99810749:@-
batch_normalization_33_99810751:@-
batch_normalization_33_99810753:@-
batch_normalization_33_99810755:@+
conv2d_3_99810758:@@
conv2d_3_99810760:@$
dense_34_99810764:	ј( 
dense_34_99810766: #
dense_35_99810769:  
dense_35_99810771:  
dense_99810774: 
dense_99810776:
identityИҐ.batch_normalization_32/StatefulPartitionedCallҐ.batch_normalization_33/StatefulPartitionedCallҐ conv2d_0/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐ dense_34/StatefulPartitionedCallҐ dense_35/StatefulPartitionedCallГ
 conv2d_0/StatefulPartitionedCallStatefulPartitionedCallinput_17conv2d_0_99810724conv2d_0_99810726*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99810198ш
 max_pooling2d_16/PartitionedCallPartitionedCall)conv2d_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99810049†
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0batch_normalization_32_99810730batch_normalization_32_99810732batch_normalization_32_99810734batch_normalization_32_99810736*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99810105≤
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_1_99810739conv2d_1_99810741*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99810225§
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_99810744conv2d_2_99810746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99810242†
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_33_99810749batch_normalization_33_99810751batch_normalization_33_99810753batch_normalization_33_99810755*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99810169≤
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_3_99810758conv2d_3_99810760*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99810268е
flatten_16/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_flatten_16_layer_call_and_return_conditional_losses_99810280Ц
 dense_34/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_34_99810764dense_34_99810766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_34_layer_call_and_return_conditional_losses_99810293Ь
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_99810769dense_35_99810771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_99810310Р
dense/StatefulPartitionedCallStatefulPartitionedCall)dense_35/StatefulPartitionedCall:output:0dense_99810774dense_99810776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_99810326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ъ
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall!^conv2d_0/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€  : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2D
 conv2d_0/StatefulPartitionedCall conv2d_0/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_17"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≤
serving_defaultЮ
E
input_179
serving_default_input_17:0€€€€€€€€€  9
dense0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:дЃ
л
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
В
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories
  _jit_compiled_convolution_op"
_tf_keras_layer
 
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
#'_self_saveable_object_factories"
_tf_keras_layer
П
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.axis
	/gamma
0beta
1moving_mean
2moving_variance
#3_self_saveable_object_factories"
_tf_keras_layer
В
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
#<_self_saveable_object_factories
 =_jit_compiled_convolution_op"
_tf_keras_layer
В
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
#F_self_saveable_object_factories
 G_jit_compiled_convolution_op"
_tf_keras_layer
П
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance
#S_self_saveable_object_factories"
_tf_keras_layer
В
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias
#\_self_saveable_object_factories
 ]_jit_compiled_convolution_op"
_tf_keras_layer
 
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
#d_self_saveable_object_factories"
_tf_keras_layer
а
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
#m_self_saveable_object_factories"
_tf_keras_layer
а
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

tkernel
ubias
#v_self_saveable_object_factories"
_tf_keras_layer
ї
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias"
_tf_keras_layer
∆
0
1
/2
03
14
25
:6
;7
D8
E9
O10
P11
Q12
R13
Z14
[15
k16
l17
t18
u19
}20
~21"
trackable_list_wrapper
¶
0
1
/2
03
:4
;5
D6
E7
O8
P9
Z10
[11
k12
l13
t14
u15
}16
~17"
trackable_list_wrapper
 "
trackable_list_wrapper
ќ
non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
щ
Дtrace_0
Еtrace_1
Жtrace_2
Зtrace_32Ж
/__inference_task_1_model_layer_call_fn_99810380
/__inference_task_1_model_layer_call_fn_99810886
/__inference_task_1_model_layer_call_fn_99810935
/__inference_task_1_model_layer_call_fn_99810662њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zДtrace_0zЕtrace_1zЖtrace_2zЗtrace_3
е
Иtrace_0
Йtrace_1
Кtrace_2
Лtrace_32т
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811018
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811101
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810721
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810780њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zИtrace_0zЙtrace_1zКtrace_2zЛtrace_3
ѕBћ
#__inference__wrapped_model_99810040input_17"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ј
	Мiter
Нbeta_1
Оbeta_2

Пdecay
Рlearning_ratemоmп/mр0mс:mт;mуDmфEmхOmцPmчZmш[mщkmъlmыtmьumэ}mю~m€vАvБ/vВ0vГ:vД;vЕDvЖEvЗOvИPvЙZvК[vЛkvМlvНtvОuvП}vР~vС"
	optimizer
-
Сserving_default"
signature_map
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
с
Чtrace_02“
+__inference_conv2d_0_layer_call_fn_99811110Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЧtrace_0
М
Шtrace_02н
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99811121Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zШtrace_0
):' 2conv2d_0/kernel
: 2conv2d_0/bias
 "
trackable_dict_wrapper
і2±Ѓ
£≤Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
щ
Юtrace_02Џ
3__inference_max_pooling2d_16_layer_call_fn_99811126Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЮtrace_0
Ф
Яtrace_02х
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99811131Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЯtrace_0
 "
trackable_dict_wrapper
<
/0
01
12
23"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
з
•trace_0
¶trace_12ђ
9__inference_batch_normalization_32_layer_call_fn_99811144
9__inference_batch_normalization_32_layer_call_fn_99811157≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z•trace_0z¶trace_1
Э
Іtrace_0
®trace_12в
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811175
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811193≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zІtrace_0z®trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_32/gamma
):' 2batch_normalization_32/beta
2:0  (2"batch_normalization_32/moving_mean
6:4  (2&batch_normalization_32/moving_variance
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
©non_trainable_variables
™layers
Ђmetrics
 ђlayer_regularization_losses
≠layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
с
Ѓtrace_02“
+__inference_conv2d_1_layer_call_fn_99811202Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЃtrace_0
М
ѓtrace_02н
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99811213Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѓtrace_0
):'  2conv2d_1/kernel
: 2conv2d_1/bias
 "
trackable_dict_wrapper
і2±Ѓ
£≤Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
∞non_trainable_variables
±layers
≤metrics
 ≥layer_regularization_losses
іlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
с
µtrace_02“
+__inference_conv2d_2_layer_call_fn_99811222Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zµtrace_0
М
ґtrace_02н
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99811233Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zґtrace_0
):' @2conv2d_2/kernel
:@2conv2d_2/bias
 "
trackable_dict_wrapper
і2±Ѓ
£≤Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Јnon_trainable_variables
Єlayers
єmetrics
 Їlayer_regularization_losses
їlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
з
Љtrace_0
љtrace_12ђ
9__inference_batch_normalization_33_layer_call_fn_99811246
9__inference_batch_normalization_33_layer_call_fn_99811259≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЉtrace_0zљtrace_1
Э
Њtrace_0
њtrace_12в
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811277
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811295≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЊtrace_0zњtrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_33/gamma
):'@2batch_normalization_33/beta
2:0@ (2"batch_normalization_33/moving_mean
6:4@ (2&batch_normalization_33/moving_variance
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
јnon_trainable_variables
Ѕlayers
¬metrics
 √layer_regularization_losses
ƒlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
с
≈trace_02“
+__inference_conv2d_3_layer_call_fn_99811304Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≈trace_0
М
∆trace_02н
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99811315Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∆trace_0
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
 "
trackable_dict_wrapper
і2±Ѓ
£≤Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
«non_trainable_variables
»layers
…metrics
  layer_regularization_losses
Ћlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
у
ћtrace_02‘
-__inference_flatten_16_layer_call_fn_99811320Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zћtrace_0
О
Ќtrace_02п
H__inference_flatten_16_layer_call_and_return_conditional_losses_99811326Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЌtrace_0
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ќnon_trainable_variables
ѕlayers
–metrics
 —layer_regularization_losses
“layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
с
”trace_02“
+__inference_dense_34_layer_call_fn_99811335Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z”trace_0
М
‘trace_02н
F__inference_dense_34_layer_call_and_return_conditional_losses_99811346Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0
": 	ј( 2dense_34/kernel
: 2dense_34/bias
 "
trackable_dict_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
’non_trainable_variables
÷layers
„metrics
 Ўlayer_regularization_losses
ўlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
с
Џtrace_02“
+__inference_dense_35_layer_call_fn_99811355Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЏtrace_0
М
џtrace_02н
F__inference_dense_35_layer_call_and_return_conditional_losses_99811366Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zџtrace_0
!:  2dense_35/kernel
: 2dense_35/bias
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
№non_trainable_variables
Ёlayers
ёmetrics
 яlayer_regularization_losses
аlayer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
о
бtrace_02ѕ
(__inference_dense_layer_call_fn_99811375Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zбtrace_0
Й
вtrace_02к
C__inference_dense_layer_call_and_return_conditional_losses_99811385Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zвtrace_0
: 2dense/kernel
:2
dense/bias
<
10
21
Q2
R3"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ВB€
/__inference_task_1_model_layer_call_fn_99810380input_17"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
/__inference_task_1_model_layer_call_fn_99810886inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
/__inference_task_1_model_layer_call_fn_99810935inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
/__inference_task_1_model_layer_call_fn_99810662input_17"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811018inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЫBШ
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811101inputs"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810721input_17"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810780input_17"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ќBЋ
&__inference_signature_wrapper_99810837input_17"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_conv2d_0_layer_call_fn_99811110inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99811121inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
3__inference_max_pooling2d_16_layer_call_fn_99811126inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99811131inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
9__inference_batch_normalization_32_layer_call_fn_99811144inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
9__inference_batch_normalization_32_layer_call_fn_99811157inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811175inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811193inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_conv2d_1_layer_call_fn_99811202inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99811213inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_conv2d_2_layer_call_fn_99811222inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99811233inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
юBы
9__inference_batch_normalization_33_layer_call_fn_99811246inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
9__inference_batch_normalization_33_layer_call_fn_99811259inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811277inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЩBЦ
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811295inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_conv2d_3_layer_call_fn_99811304inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99811315inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
бBё
-__inference_flatten_16_layer_call_fn_99811320inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ьBщ
H__inference_flatten_16_layer_call_and_return_conditional_losses_99811326inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_dense_34_layer_call_fn_99811335inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_dense_34_layer_call_and_return_conditional_losses_99811346inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB№
+__inference_dense_35_layer_call_fn_99811355inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ъBч
F__inference_dense_35_layer_call_and_return_conditional_losses_99811366inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_layer_call_fn_99811375inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_layer_call_and_return_conditional_losses_99811385inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
е	variables
ж	keras_api

зtotal

иcount"
_tf_keras_metric
c
й	variables
к	keras_api

лtotal

мcount
н
_fn_kwargs"
_tf_keras_metric
0
з0
и1"
trackable_list_wrapper
.
е	variables"
_generic_user_object
:  (2total
:  (2count
0
л0
м1"
trackable_list_wrapper
.
й	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.:, 2Adam/conv2d_0/kernel/m
 : 2Adam/conv2d_0/bias/m
/:- 2#Adam/batch_normalization_32/gamma/m
.:, 2"Adam/batch_normalization_32/beta/m
.:,  2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, @2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
/:-@2#Adam/batch_normalization_33/gamma/m
.:,@2"Adam/batch_normalization_33/beta/m
.:,@@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
':%	ј( 2Adam/dense_34/kernel/m
 : 2Adam/dense_34/bias/m
&:$  2Adam/dense_35/kernel/m
 : 2Adam/dense_35/bias/m
#:! 2Adam/dense/kernel/m
:2Adam/dense/bias/m
.:, 2Adam/conv2d_0/kernel/v
 : 2Adam/conv2d_0/bias/v
/:- 2#Adam/batch_normalization_32/gamma/v
.:, 2"Adam/batch_normalization_32/beta/v
.:,  2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, @2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
/:-@2#Adam/batch_normalization_33/gamma/v
.:,@2"Adam/batch_normalization_33/beta/v
.:,@@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
':%	ј( 2Adam/dense_34/kernel/v
 : 2Adam/dense_34/bias/v
&:$  2Adam/dense_35/kernel/v
 : 2Adam/dense_35/bias/v
#:! 2Adam/dense/kernel/v
:2Adam/dense/bias/v™
#__inference__wrapped_model_99810040В/012:;DEOPQRZ[kltu}~9Ґ6
/Ґ,
*К'
input_17€€€€€€€€€  
™ "-™*
(
denseК
dense€€€€€€€€€п
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811175Ц/012MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ п
T__inference_batch_normalization_32_layer_call_and_return_conditional_losses_99811193Ц/012MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ «
9__inference_batch_normalization_32_layer_call_fn_99811144Й/012MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ «
9__inference_batch_normalization_32_layer_call_fn_99811157Й/012MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ п
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811277ЦOPQRMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ п
T__inference_batch_normalization_33_layer_call_and_return_conditional_losses_99811295ЦOPQRMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ «
9__inference_batch_normalization_33_layer_call_fn_99811246ЙOPQRMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@«
9__inference_batch_normalization_33_layer_call_fn_99811259ЙOPQRMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ґ
F__inference_conv2d_0_layer_call_and_return_conditional_losses_99811121l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ О
+__inference_conv2d_0_layer_call_fn_99811110_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  
™ " К€€€€€€€€€ ґ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_99811213l:;7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ О
+__inference_conv2d_1_layer_call_fn_99811202_:;7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ " К€€€€€€€€€ ґ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_99811233lDE7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ О
+__inference_conv2d_2_layer_call_fn_99811222_DE7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ " К€€€€€€€€€@ґ
F__inference_conv2d_3_layer_call_and_return_conditional_losses_99811315lZ[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ "-Ґ*
#К 
0€€€€€€€€€		@
Ъ О
+__inference_conv2d_3_layer_call_fn_99811304_Z[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ " К€€€€€€€€€		@І
F__inference_dense_34_layer_call_and_return_conditional_losses_99811346]kl0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј(
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ 
+__inference_dense_34_layer_call_fn_99811335Pkl0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј(
™ "К€€€€€€€€€ ¶
F__inference_dense_35_layer_call_and_return_conditional_losses_99811366\tu/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ~
+__inference_dense_35_layer_call_fn_99811355Otu/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€ £
C__inference_dense_layer_call_and_return_conditional_losses_99811385\}~/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_layer_call_fn_99811375O}~/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€≠
H__inference_flatten_16_layer_call_and_return_conditional_losses_99811326a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€		@
™ "&Ґ#
К
0€€€€€€€€€ј(
Ъ Е
-__inference_flatten_16_layer_call_fn_99811320T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€		@
™ "К€€€€€€€€€ј(с
N__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_99811131ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ …
3__inference_max_pooling2d_16_layer_call_fn_99811126СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€є
&__inference_signature_wrapper_99810837О/012:;DEOPQRZ[kltu}~EҐB
Ґ 
;™8
6
input_17*К'
input_17€€€€€€€€€  "-™*
(
denseК
dense€€€€€€€€€—
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810721В/012:;DEOPQRZ[kltu}~AҐ>
7Ґ4
*К'
input_17€€€€€€€€€  
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ —
J__inference_task_1_model_layer_call_and_return_conditional_losses_99810780В/012:;DEOPQRZ[kltu}~AҐ>
7Ґ4
*К'
input_17€€€€€€€€€  
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ѕ
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811018А/012:;DEOPQRZ[kltu}~?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ѕ
J__inference_task_1_model_layer_call_and_return_conditional_losses_99811101А/012:;DEOPQRZ[kltu}~?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ®
/__inference_task_1_model_layer_call_fn_99810380u/012:;DEOPQRZ[kltu}~AҐ>
7Ґ4
*К'
input_17€€€€€€€€€  
p 

 
™ "К€€€€€€€€€®
/__inference_task_1_model_layer_call_fn_99810662u/012:;DEOPQRZ[kltu}~AҐ>
7Ґ4
*К'
input_17€€€€€€€€€  
p

 
™ "К€€€€€€€€€¶
/__inference_task_1_model_layer_call_fn_99810886s/012:;DEOPQRZ[kltu}~?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p 

 
™ "К€€€€€€€€€¶
/__inference_task_1_model_layer_call_fn_99810935s/012:;DEOPQRZ[kltu}~?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p

 
™ "К€€€€€€€€€