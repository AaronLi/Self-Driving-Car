
¨ù
.
Abs
x"T
y"T"
Ttype:

2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
º
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ù
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	

sequential_2/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namesequential_2/dense/kernel

-sequential_2/dense/kernel/Read/ReadVariableOpReadVariableOpsequential_2/dense/kernel* 
_output_shapes
:
*
dtype0

sequential_2/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential_2/dense/bias

+sequential_2/dense/bias/Read/ReadVariableOpReadVariableOpsequential_2/dense/bias*
_output_shapes	
:*
dtype0

sequential_2/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namesequential_2/dense_1/kernel

/sequential_2/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential_2/dense_1/kernel* 
_output_shapes
:
*
dtype0

sequential_2/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namesequential_2/dense_1/bias

-sequential_2/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential_2/dense_1/bias*
_output_shapes	
:*
dtype0

sequential_2/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*,
shared_namesequential_2/dense_2/kernel

/sequential_2/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential_2/dense_2/kernel*
_output_shapes
:	@*
dtype0

sequential_2/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namesequential_2/dense_2/bias

-sequential_2/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential_2/dense_2/bias*
_output_shapes
:@*
dtype0

sequential_2/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *,
shared_namesequential_2/dense_3/kernel

/sequential_2/dense_3/kernel/Read/ReadVariableOpReadVariableOpsequential_2/dense_3/kernel*
_output_shapes

:@ *
dtype0

sequential_2/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namesequential_2/dense_3/bias

-sequential_2/dense_3/bias/Read/ReadVariableOpReadVariableOpsequential_2/dense_3/bias*
_output_shapes
: *
dtype0
°
*sequential_2/means_projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*sequential_2/means_projection_layer/kernel
©
>sequential_2/means_projection_layer/kernel/Read/ReadVariableOpReadVariableOp*sequential_2/means_projection_layer/kernel*
_output_shapes

: *
dtype0
¨
(sequential_2/means_projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sequential_2/means_projection_layer/bias
¡
<sequential_2/means_projection_layer/bias/Read/ReadVariableOpReadVariableOp(sequential_2/means_projection_layer/bias*
_output_shapes
:*
dtype0
¼
2sequential_2/nest_map/sequential_1/bias_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42sequential_2/nest_map/sequential_1/bias_layer/bias
µ
Fsequential_2/nest_map/sequential_1/bias_layer/bias/Read/ReadVariableOpReadVariableOp2sequential_2/nest_map/sequential_1/bias_layer/bias*
_output_shapes
:*
dtype0
c
avg_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameavg_0
\
avg_0/Read/ReadVariableOpReadVariableOpavg_0*
_output_shapes	
:*
dtype0
g
count_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	count_0
`
count_0/Read/ReadVariableOpReadVariableOpcount_0*
_output_shapes	
:*
dtype0
a
m2_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namem2_0
Z
m2_0/Read/ReadVariableOpReadVariableOpm2_0*
_output_shapes	
:*
dtype0
m

m2_carry_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
m2_carry_0
f
m2_carry_0/Read/ReadVariableOpReadVariableOp
m2_carry_0*
_output_shapes	
:*
dtype0
³
+ValueNetwork/EncodingNetwork/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K*<
shared_name-+ValueNetwork/EncodingNetwork/dense_4/kernel
¬
?ValueNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOpReadVariableOp+ValueNetwork/EncodingNetwork/dense_4/kernel*
_output_shapes
:	K*
dtype0
ª
)ValueNetwork/EncodingNetwork/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:K*:
shared_name+)ValueNetwork/EncodingNetwork/dense_4/bias
£
=ValueNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOpReadVariableOp)ValueNetwork/EncodingNetwork/dense_4/bias*
_output_shapes
:K*
dtype0
²
+ValueNetwork/EncodingNetwork/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:K(*<
shared_name-+ValueNetwork/EncodingNetwork/dense_5/kernel
«
?ValueNetwork/EncodingNetwork/dense_5/kernel/Read/ReadVariableOpReadVariableOp+ValueNetwork/EncodingNetwork/dense_5/kernel*
_output_shapes

:K(*
dtype0
ª
)ValueNetwork/EncodingNetwork/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*:
shared_name+)ValueNetwork/EncodingNetwork/dense_5/bias
£
=ValueNetwork/EncodingNetwork/dense_5/bias/Read/ReadVariableOpReadVariableOp)ValueNetwork/EncodingNetwork/dense_5/bias*
_output_shapes
:(*
dtype0

ValueNetwork/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*,
shared_nameValueNetwork/dense_6/kernel

/ValueNetwork/dense_6/kernel/Read/ReadVariableOpReadVariableOpValueNetwork/dense_6/kernel*
_output_shapes

:(*
dtype0

ValueNetwork/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameValueNetwork/dense_6/bias

-ValueNetwork/dense_6/bias/Read/ReadVariableOpReadVariableOpValueNetwork/dense_6/bias*
_output_shapes
:*
dtype0

NoOpNoOp
îh
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*©h
valuehBh Bh
³

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures*
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
¢
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20*

 _wrapped_policy*
* 
* 
* 
* 
* 
K

!action
"get_initial_state
#get_train_step
$get_metadata* 
_Y
VARIABLE_VALUEsequential_2/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEsequential_2/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEsequential_2/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential_2/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEsequential_2/dense_2/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential_2/dense_2/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEsequential_2/dense_3/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential_2/dense_3/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE*sequential_2/means_projection_layer/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE(sequential_2/means_projection_layer/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE2sequential_2/nest_map/sequential_1/bias_layer/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEavg_0-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEcount_0-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEm2_0-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUE
m2_carry_0-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE+ValueNetwork/EncodingNetwork/dense_4/kernel-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE)ValueNetwork/EncodingNetwork/dense_4/bias-model_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE+ValueNetwork/EncodingNetwork/dense_5/kernel-model_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE)ValueNetwork/EncodingNetwork/dense_5/bias-model_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEValueNetwork/dense_6/kernel-model_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEValueNetwork/dense_6/bias-model_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
I
%_actor_network
&_observation_normalizer
'_value_network*
* 
* 
* 
* 
Ø
(_layer_state_is_list
)_sequential_layers
*_layer_has_state
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
K
1_flat_variable_spec

2_count
3_avg
4_m2
5	_m2_carry*
º
6_encoder
7_postprocessing_layers
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
* 
<
>0
?1
@2
A3
B4
C5
D6
E7*
* 
R
0
1
2
3
4
5
6
7
8
9
10*
R
0
1
2
3
4
5
6
7
8
9
10*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 

0*

0*

0*

0*
¬
K_postprocessing_layers
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses*
¦

kernel
bias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 

Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
¦

kernel
bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses*
¦

kernel
bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses*
¦

kernel
bias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
¦

kernel
bias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*
¦

kernel
bias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses*

{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses* 
½
_state_spec
_nested_layers
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 
<
>0
?1
@2
A3
B4
C5
D6
E7*
* 
* 
* 

0
1
2*
 
0
1
2
3*
 
0
1
2
3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
* 

60
71*
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 

ºloc

»scale*

0*

0*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses* 
¬

kernel
bias
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses*
¬

kernel
bias
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses*
* 

0
1
2*
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
ß
Ø_layer_state_is_list
Ù_sequential_layers
Ú_layer_has_state
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses* 
á
á_layer_state_is_list
â_sequential_layers
ã_layer_has_state
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses*
* 

º0
»1*
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

ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses*
* 
* 
* 


ù0* 
* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses* 
* 
* 
* 

ÿ0*
* 

0*

0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses*
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

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 


ù0* 
* 
* 
* 
 
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
* 

ÿ0*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

0*

0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
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
l
action_0_discountPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
y
action_0_observationPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
j
action_0_rewardPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
m
action_0_step_typePlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_typem2_0count_0avg_0sequential_2/dense/kernelsequential_2/dense/biassequential_2/dense_1/kernelsequential_2/dense_1/biassequential_2/dense_2/kernelsequential_2/dense_2/biassequential_2/dense_3/kernelsequential_2/dense_3/bias*sequential_2/means_projection_layer/kernel(sequential_2/means_projection_layer/bias2sequential_2/nest_map/sequential_1/bias_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_92040619
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
û
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_92040631
Ü
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_92040653

StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_92040646
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
å

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp-sequential_2/dense/kernel/Read/ReadVariableOp+sequential_2/dense/bias/Read/ReadVariableOp/sequential_2/dense_1/kernel/Read/ReadVariableOp-sequential_2/dense_1/bias/Read/ReadVariableOp/sequential_2/dense_2/kernel/Read/ReadVariableOp-sequential_2/dense_2/bias/Read/ReadVariableOp/sequential_2/dense_3/kernel/Read/ReadVariableOp-sequential_2/dense_3/bias/Read/ReadVariableOp>sequential_2/means_projection_layer/kernel/Read/ReadVariableOp<sequential_2/means_projection_layer/bias/Read/ReadVariableOpFsequential_2/nest_map/sequential_1/bias_layer/bias/Read/ReadVariableOpavg_0/Read/ReadVariableOpcount_0/Read/ReadVariableOpm2_0/Read/ReadVariableOpm2_carry_0/Read/ReadVariableOp?ValueNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOp=ValueNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOp?ValueNetwork/EncodingNetwork/dense_5/kernel/Read/ReadVariableOp=ValueNetwork/EncodingNetwork/dense_5/bias/Read/ReadVariableOp/ValueNetwork/dense_6/kernel/Read/ReadVariableOp-ValueNetwork/dense_6/bias/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_92041323
¨
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariablesequential_2/dense/kernelsequential_2/dense/biassequential_2/dense_1/kernelsequential_2/dense_1/biassequential_2/dense_2/kernelsequential_2/dense_2/biassequential_2/dense_3/kernelsequential_2/dense_3/bias*sequential_2/means_projection_layer/kernel(sequential_2/means_projection_layer/bias2sequential_2/nest_map/sequential_1/bias_layer/biasavg_0count_0m2_0
m2_carry_0+ValueNetwork/EncodingNetwork/dense_4/kernel)ValueNetwork/EncodingNetwork/dense_4/bias+ValueNetwork/EncodingNetwork/dense_5/kernel)ValueNetwork/EncodingNetwork/dense_5/biasValueNetwork/dense_6/kernelValueNetwork/dense_6/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_92041399âÏ
ê>
ë
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92040950ä
ßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ú
Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constØ
Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_absÃ
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
¢ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert¶
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.­
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:ª
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = ¨
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = °
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/AssertAssertßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_allÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2:output:0Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4:output:0Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_abs*
T

2*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ð
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¹
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ2ú
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assertºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦

,__inference_function_with_signature_92040581
	step_type

reward
discount
observation
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	
	unknown_6:	@
	unknown_7:@
	unknown_8:@ 
	unknown_9: 

unknown_10: 

unknown_11:

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *3
f.R,
*__inference_polymorphic_action_fn_92040550o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
0/discount:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_name0/observation
5

!__inference__traced_save_92041323
file_prefix'
#savev2_variable_read_readvariableop	8
4savev2_sequential_2_dense_kernel_read_readvariableop6
2savev2_sequential_2_dense_bias_read_readvariableop:
6savev2_sequential_2_dense_1_kernel_read_readvariableop8
4savev2_sequential_2_dense_1_bias_read_readvariableop:
6savev2_sequential_2_dense_2_kernel_read_readvariableop8
4savev2_sequential_2_dense_2_bias_read_readvariableop:
6savev2_sequential_2_dense_3_kernel_read_readvariableop8
4savev2_sequential_2_dense_3_bias_read_readvariableopI
Esavev2_sequential_2_means_projection_layer_kernel_read_readvariableopG
Csavev2_sequential_2_means_projection_layer_bias_read_readvariableopQ
Msavev2_sequential_2_nest_map_sequential_1_bias_layer_bias_read_readvariableop$
 savev2_avg_0_read_readvariableop&
"savev2_count_0_read_readvariableop#
savev2_m2_0_read_readvariableop)
%savev2_m2_carry_0_read_readvariableopJ
Fsavev2_valuenetwork_encodingnetwork_dense_4_kernel_read_readvariableopH
Dsavev2_valuenetwork_encodingnetwork_dense_4_bias_read_readvariableopJ
Fsavev2_valuenetwork_encodingnetwork_dense_5_kernel_read_readvariableopH
Dsavev2_valuenetwork_encodingnetwork_dense_5_bias_read_readvariableop:
6savev2_valuenetwork_dense_6_kernel_read_readvariableop8
4savev2_valuenetwork_dense_6_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*«
value¡BB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop4savev2_sequential_2_dense_kernel_read_readvariableop2savev2_sequential_2_dense_bias_read_readvariableop6savev2_sequential_2_dense_1_kernel_read_readvariableop4savev2_sequential_2_dense_1_bias_read_readvariableop6savev2_sequential_2_dense_2_kernel_read_readvariableop4savev2_sequential_2_dense_2_bias_read_readvariableop6savev2_sequential_2_dense_3_kernel_read_readvariableop4savev2_sequential_2_dense_3_bias_read_readvariableopEsavev2_sequential_2_means_projection_layer_kernel_read_readvariableopCsavev2_sequential_2_means_projection_layer_bias_read_readvariableopMsavev2_sequential_2_nest_map_sequential_1_bias_layer_bias_read_readvariableop savev2_avg_0_read_readvariableop"savev2_count_0_read_readvariableopsavev2_m2_0_read_readvariableop%savev2_m2_carry_0_read_readvariableopFsavev2_valuenetwork_encodingnetwork_dense_4_kernel_read_readvariableopDsavev2_valuenetwork_encodingnetwork_dense_4_bias_read_readvariableopFsavev2_valuenetwork_encodingnetwork_dense_5_kernel_read_readvariableopDsavev2_valuenetwork_encodingnetwork_dense_5_bias_read_readvariableop6savev2_valuenetwork_dense_6_kernel_read_readvariableop4savev2_valuenetwork_dense_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*Å
_input_shapes³
°: : :
::
::	@:@:@ : : :::::::	K:K:K(:(:(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 	

_output_shapes
: :$
 

_output_shapes

: : 

_output_shapes
:: 

_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	K: 

_output_shapes
:K:$ 

_output_shapes

:K(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: 
ó
	
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92040448æ
ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ä
¿sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholderÆ
Ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholder_1Ã
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
×
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ´
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
áð
Ø
*__inference_polymorphic_action_fn_92041051
time_step_step_type
time_step_reward
time_step_discount
time_step_observationG
8normalize_observations_normalize_readvariableop_resource:	O
@normalize_observations_normalize_truediv_readvariableop_resource:	Y
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resource:	E
1sequential_2_dense_matmul_readvariableop_resource:
A
2sequential_2_dense_biasadd_readvariableop_resource:	G
3sequential_2_dense_1_matmul_readvariableop_resource:
C
4sequential_2_dense_1_biasadd_readvariableop_resource:	F
3sequential_2_dense_2_matmul_readvariableop_resource:	@B
4sequential_2_dense_2_biasadd_readvariableop_resource:@E
3sequential_2_dense_3_matmul_readvariableop_resource:@ B
4sequential_2_dense_3_biasadd_readvariableop_resource: T
Bsequential_2_means_projection_layer_matmul_readvariableop_resource: Q
Csequential_2_means_projection_layer_biasadd_readvariableop_resource:[
Msequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource:
identity¢/normalize_observations/normalize/ReadVariableOp¢Anormalize_observations/normalize/normalized_tensor/ReadVariableOp¢7normalize_observations/normalize/truediv/ReadVariableOp¢)sequential_2/dense/BiasAdd/ReadVariableOp¢(sequential_2/dense/MatMul/ReadVariableOp¢+sequential_2/dense_1/BiasAdd/ReadVariableOp¢*sequential_2/dense_1/MatMul/ReadVariableOp¢+sequential_2/dense_2/BiasAdd/ReadVariableOp¢*sequential_2/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard¢:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp¢9sequential_2/means_projection_layer/MatMul/ReadVariableOp¢Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp¥
/normalize_observations/normalize/ReadVariableOpReadVariableOp8normalize_observations_normalize_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7normalize_observations/normalize/truediv/ReadVariableOpReadVariableOp@normalize_observations_normalize_truediv_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
(normalize_observations/normalize/truedivRealDiv7normalize_observations/normalize/ReadVariableOp:value:0?normalize_observations/normalize/truediv/ReadVariableOp:value:0*
T0*
_output_shapes	
:}
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ö
6normalize_observations/normalize/normalized_tensor/addAddV2,normalize_observations/normalize/truediv:z:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes	
:£
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes	
:Å
6normalize_observations/normalize/normalized_tensor/mulMultime_step_observation<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes	
:*
dtype0®
6normalize_observations/normalize/normalized_tensor/NegNegInormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes	
:î
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *   À
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_2/dense/MatMul/ReadVariableOpReadVariableOp1sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0È
sequential_2/dense/MatMulMatMul>normalize_observations/normalize/clipped_normalized_tensor:z:00sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential_2/dense/BiasAddBiasAdd#sequential_2/dense/MatMul:product:01sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential_2/dense/TanhTanh#sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
sequential_2/dense_1/MatMulMatMulsequential_2/dense/Tanh:y:02sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_2/dense_1/BiasAddBiasAdd%sequential_2/dense_1/MatMul:product:03sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_2/dense_1/TanhTanh%sequential_2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0ª
sequential_2/dense_2/MatMulMatMulsequential_2/dense_1/Tanh:y:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_2/TanhTanh%sequential_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0ª
sequential_2/dense_3/MatMulMatMulsequential_2/dense_2/Tanh:y:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
sequential_2/dense_3/TanhTanh%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
9sequential_2/means_projection_layer/MatMul/ReadVariableOpReadVariableOpBsequential_2_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0È
*sequential_2/means_projection_layer/MatMulMatMulsequential_2/dense_3/Tanh:y:0Asequential_2/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpCsequential_2_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0â
+sequential_2/means_projection_layer/BiasAddBiasAdd4sequential_2/means_projection_layer/MatMul:product:0Bsequential_2/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda/zeros_like	ZerosLike4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpReadVariableOpMsequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ä
5sequential_2/nest_map/sequential_1/bias_layer/BiasAddBiasAdd"sequential_2/lambda/zeros_like:y:0Lsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/TanhTanh4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/mulMul$sequential_2/lambda_2/mul/x:output:0sequential_2/lambda_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/addAddV2$sequential_2/lambda_2/add/x:output:0sequential_2/lambda_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/SoftplusSoftplus>sequential_2/nest_map/sequential_1/bias_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :
Rsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:ï
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:ô
 sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿí
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: í
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0©sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskù
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack£sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:Û
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¸
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0¥sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0¡sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ö
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿÍ
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:¤
Zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¯
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¦
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:²
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceUsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0csequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskæ
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgssequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0]sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/AbsAbs,sequential_2/lambda_2/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
¥sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/LessLess£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
¦sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¢
¤sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/AllAll©sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Less:z:0¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Const:output:0*
_output_shapes
: ¢
­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = 
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_3Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = ²
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuardIf­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ø
else_branchÈRÅ
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92040950*
output_shapes
: *×
then_branchÇRÄ
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92040949©
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentity¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: ²
Tsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/group_depsNoOp½^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity*
_output_shapes
 }
8sequential_2/lambda_2/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2sequential_2/lambda_2/MultivariateNormalDiag/zerosFillYsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0Asequential_2/lambda_2/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
1sequential_2/lambda_2/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?s
1sequential_2/lambda_2/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ShapeShape;sequential_2/lambda_2/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:í
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
¸MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_sliceStridedSlice³MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Shape:output:0ÁMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskø
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB ï
¬MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1StridedSlice½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensor:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskù
µMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB û
·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB á
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsÀMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0»MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:Ü
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgs·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs:r0:0½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:
°MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shapeIdentity¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:Ó
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : é
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_sliceStridedSlice¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shape:output:0§MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskß
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB á
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgsBroadcastArgs¦MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1:output:0¡MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:è
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:½
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:²
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :È
~MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskÎ
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1Shape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:´
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSlice{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask¾
{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB À
}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgsMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:¬
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgs}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:¨
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:º
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:¢
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shapeIdentityyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ú
GMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concatConcatV2MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shape:output:0UMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:þ
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastToBroadcastTosequential_2/lambda_2/add:z:0PMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Deterministic/sample/ShapeShapeUMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastTo:output:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:â
 Deterministic/sample/BroadcastToBroadcastToUMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastTo:output:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¬
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
NoOpNoOp0^normalize_observations/normalize/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOp8^normalize_observations/normalize/truediv/ReadVariableOp*^sequential_2/dense/BiasAdd/ReadVariableOp)^sequential_2/dense/MatMul/ReadVariableOp,^sequential_2/dense_1/BiasAdd/ReadVariableOp+^sequential_2/dense_1/MatMul/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp´^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard;^sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:^sequential_2/means_projection_layer/MatMul/ReadVariableOpE^sequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/normalize_observations/normalize/ReadVariableOp/normalize_observations/normalize/ReadVariableOp2
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2r
7normalize_observations/normalize/truediv/ReadVariableOp7normalize_observations/normalize/truediv/ReadVariableOp2V
)sequential_2/dense/BiasAdd/ReadVariableOp)sequential_2/dense/BiasAdd/ReadVariableOp2T
(sequential_2/dense/MatMul/ReadVariableOp(sequential_2/dense/MatMul/ReadVariableOp2Z
+sequential_2/dense_1/BiasAdd/ReadVariableOp+sequential_2/dense_1/BiasAdd/ReadVariableOp2X
*sequential_2/dense_1/MatMul/ReadVariableOp*sequential_2/dense_1/MatMul/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2ì
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard2x
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp2v
9sequential_2/means_projection_layer/MatMul/ReadVariableOp9sequential_2/means_projection_layer/MatMul/ReadVariableOp2
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpDsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:X T
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_nametime_step/discount:_[
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_nametime_step/observation
ÔÓ
Ö
0__inference_polymorphic_distribution_fn_92041225
	step_type

reward
discount
observationG
8normalize_observations_normalize_readvariableop_resource:	O
@normalize_observations_normalize_truediv_readvariableop_resource:	Y
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resource:	E
1sequential_2_dense_matmul_readvariableop_resource:
A
2sequential_2_dense_biasadd_readvariableop_resource:	G
3sequential_2_dense_1_matmul_readvariableop_resource:
C
4sequential_2_dense_1_biasadd_readvariableop_resource:	F
3sequential_2_dense_2_matmul_readvariableop_resource:	@B
4sequential_2_dense_2_biasadd_readvariableop_resource:@E
3sequential_2_dense_3_matmul_readvariableop_resource:@ B
4sequential_2_dense_3_biasadd_readvariableop_resource: T
Bsequential_2_means_projection_layer_matmul_readvariableop_resource: Q
Csequential_2_means_projection_layer_biasadd_readvariableop_resource:[
Msequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢/normalize_observations/normalize/ReadVariableOp¢Anormalize_observations/normalize/normalized_tensor/ReadVariableOp¢7normalize_observations/normalize/truediv/ReadVariableOp¢)sequential_2/dense/BiasAdd/ReadVariableOp¢(sequential_2/dense/MatMul/ReadVariableOp¢+sequential_2/dense_1/BiasAdd/ReadVariableOp¢*sequential_2/dense_1/MatMul/ReadVariableOp¢+sequential_2/dense_2/BiasAdd/ReadVariableOp¢*sequential_2/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard¢:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp¢9sequential_2/means_projection_layer/MatMul/ReadVariableOp¢Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp¥
/normalize_observations/normalize/ReadVariableOpReadVariableOp8normalize_observations_normalize_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7normalize_observations/normalize/truediv/ReadVariableOpReadVariableOp@normalize_observations_normalize_truediv_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
(normalize_observations/normalize/truedivRealDiv7normalize_observations/normalize/ReadVariableOp:value:0?normalize_observations/normalize/truediv/ReadVariableOp:value:0*
T0*
_output_shapes	
:}
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ö
6normalize_observations/normalize/normalized_tensor/addAddV2,normalize_observations/normalize/truediv:z:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes	
:£
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes	
:»
6normalize_observations/normalize/normalized_tensor/mulMulobservation<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes	
:*
dtype0®
6normalize_observations/normalize/normalized_tensor/NegNegInormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes	
:î
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *   À
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_2/dense/MatMul/ReadVariableOpReadVariableOp1sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0È
sequential_2/dense/MatMulMatMul>normalize_observations/normalize/clipped_normalized_tensor:z:00sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential_2/dense/BiasAddBiasAdd#sequential_2/dense/MatMul:product:01sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential_2/dense/TanhTanh#sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
sequential_2/dense_1/MatMulMatMulsequential_2/dense/Tanh:y:02sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_2/dense_1/BiasAddBiasAdd%sequential_2/dense_1/MatMul:product:03sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_2/dense_1/TanhTanh%sequential_2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0ª
sequential_2/dense_2/MatMulMatMulsequential_2/dense_1/Tanh:y:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_2/TanhTanh%sequential_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0ª
sequential_2/dense_3/MatMulMatMulsequential_2/dense_2/Tanh:y:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
sequential_2/dense_3/TanhTanh%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
9sequential_2/means_projection_layer/MatMul/ReadVariableOpReadVariableOpBsequential_2_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0È
*sequential_2/means_projection_layer/MatMulMatMulsequential_2/dense_3/Tanh:y:0Asequential_2/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpCsequential_2_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0â
+sequential_2/means_projection_layer/BiasAddBiasAdd4sequential_2/means_projection_layer/MatMul:product:0Bsequential_2/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda/zeros_like	ZerosLike4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpReadVariableOpMsequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ä
5sequential_2/nest_map/sequential_1/bias_layer/BiasAddBiasAdd"sequential_2/lambda/zeros_like:y:0Lsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/TanhTanh4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/mulMul$sequential_2/lambda_2/mul/x:output:0sequential_2/lambda_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/addAddV2$sequential_2/lambda_2/add/x:output:0sequential_2/lambda_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/SoftplusSoftplus>sequential_2/nest_map/sequential_1/bias_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :
Rsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:ï
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:ô
 sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿí
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: í
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0©sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskù
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack£sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:Û
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¸
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0¥sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0¡sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ö
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿÍ
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:¤
Zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¯
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¦
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:²
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceUsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0csequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskæ
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgssequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0]sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/AbsAbs,sequential_2/lambda_2/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
¥sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/LessLess£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
¦sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¢
¤sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/AllAll©sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Less:z:0¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Const:output:0*
_output_shapes
: ¢
­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = 
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_3Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = ²
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuardIf­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ø
else_branchÈRÅ
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92041149*
output_shapes
: *×
then_branchÇRÄ
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92041148©
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentity¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: ²
Tsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/group_depsNoOp½^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity*
_output_shapes
 }
8sequential_2/lambda_2/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2sequential_2/lambda_2/MultivariateNormalDiag/zerosFillYsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0Asequential_2/lambda_2/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
1sequential_2/lambda_2/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?s
1sequential_2/lambda_2/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ShapeShape;sequential_2/lambda_2/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:í
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
¸MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_sliceStridedSlice³MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Shape:output:0ÁMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskø
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB ï
¬MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1StridedSlice½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensor:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskù
µMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB û
·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB á
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsÀMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0»MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:Ü
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgs·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs:r0:0½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:
°MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shapeIdentity¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:Ó
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : é
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_sliceStridedSlice¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shape:output:0§MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskß
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB á
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgsBroadcastArgs¦MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1:output:0¡MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:è
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:½
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:²
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :È
~MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskÎ
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1Shape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:´
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSlice{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask¾
{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB À
}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgsMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:¬
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgs}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:¨
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:º
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:¢
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shapeIdentityyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ú
GMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concatConcatV2MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shape:output:0UMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:þ
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastToBroadcastTosequential_2/lambda_2/add:z:0PMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: ¦

Identity_1IdentityUMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastTo:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: Õ
NoOpNoOp0^normalize_observations/normalize/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOp8^normalize_observations/normalize/truediv/ReadVariableOp*^sequential_2/dense/BiasAdd/ReadVariableOp)^sequential_2/dense/MatMul/ReadVariableOp,^sequential_2/dense_1/BiasAdd/ReadVariableOp+^sequential_2/dense_1/MatMul/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp´^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard;^sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:^sequential_2/means_projection_layer/MatMul/ReadVariableOpE^sequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/normalize_observations/normalize/ReadVariableOp/normalize_observations/normalize/ReadVariableOp2
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2r
7normalize_observations/normalize/truediv/ReadVariableOp7normalize_observations/normalize/truediv/ReadVariableOp2V
)sequential_2/dense/BiasAdd/ReadVariableOp)sequential_2/dense/BiasAdd/ReadVariableOp2T
(sequential_2/dense/MatMul/ReadVariableOp(sequential_2/dense/MatMul/ReadVariableOp2Z
+sequential_2/dense_1/BiasAdd/ReadVariableOp+sequential_2/dense_1/BiasAdd/ReadVariableOp2X
*sequential_2/dense_1/MatMul/ReadVariableOp*sequential_2/dense_1/MatMul/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2ì
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard2x
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp2v
9sequential_2/means_projection_layer/MatMul/ReadVariableOp9sequential_2/means_projection_layer/MatMul/ReadVariableOp2
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpDsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	step_type:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namereward:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
discount:UQ
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameobservation
¿
8
&__inference_get_initial_state_92041228

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ó
	
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92041148æ
ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ä
¿sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholderÆ
Ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholder_1Ã
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
×
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ´
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
÷
d
__inference_<lambda>_92039967!
readvariableop_resource:	 
identity	¢ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
Ç
8
&__inference_signature_wrapper_92040631

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_92040626*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ê>
ë
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92040751ä
ßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ú
Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constØ
Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_absÃ
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
¢ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert¶
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.­
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:ª
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = ¨
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = °
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/AssertAssertßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_allÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2:output:0Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4:output:0Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_abs*
T

2*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ð
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¹
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ2ú
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assertºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÁZ
¥
$__inference__traced_restore_92041399
file_prefix#
assignvariableop_variable:	 @
,assignvariableop_1_sequential_2_dense_kernel:
9
*assignvariableop_2_sequential_2_dense_bias:	B
.assignvariableop_3_sequential_2_dense_1_kernel:
;
,assignvariableop_4_sequential_2_dense_1_bias:	A
.assignvariableop_5_sequential_2_dense_2_kernel:	@:
,assignvariableop_6_sequential_2_dense_2_bias:@@
.assignvariableop_7_sequential_2_dense_3_kernel:@ :
,assignvariableop_8_sequential_2_dense_3_bias: O
=assignvariableop_9_sequential_2_means_projection_layer_kernel: J
<assignvariableop_10_sequential_2_means_projection_layer_bias:T
Fassignvariableop_11_sequential_2_nest_map_sequential_1_bias_layer_bias:(
assignvariableop_12_avg_0:	*
assignvariableop_13_count_0:	'
assignvariableop_14_m2_0:	-
assignvariableop_15_m2_carry_0:	R
?assignvariableop_16_valuenetwork_encodingnetwork_dense_4_kernel:	KK
=assignvariableop_17_valuenetwork_encodingnetwork_dense_4_bias:KQ
?assignvariableop_18_valuenetwork_encodingnetwork_dense_5_kernel:K(K
=assignvariableop_19_valuenetwork_encodingnetwork_dense_5_bias:(A
/assignvariableop_20_valuenetwork_dense_6_kernel:(;
-assignvariableop_21_valuenetwork_dense_6_bias:
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*«
value¡BB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/16/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/17/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/18/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/19/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/20/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp,assignvariableop_1_sequential_2_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp*assignvariableop_2_sequential_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_sequential_2_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp,assignvariableop_4_sequential_2_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp.assignvariableop_5_sequential_2_dense_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_sequential_2_dense_2_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp.assignvariableop_7_sequential_2_dense_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp,assignvariableop_8_sequential_2_dense_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_9AssignVariableOp=assignvariableop_9_sequential_2_means_projection_layer_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_10AssignVariableOp<assignvariableop_10_sequential_2_means_projection_layer_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_11AssignVariableOpFassignvariableop_11_sequential_2_nest_map_sequential_1_bias_layer_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_avg_0Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_0Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_m2_0Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_m2_carry_0Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_16AssignVariableOp?assignvariableop_16_valuenetwork_encodingnetwork_dense_4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_17AssignVariableOp=assignvariableop_17_valuenetwork_encodingnetwork_dense_4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_18AssignVariableOp?assignvariableop_18_valuenetwork_encodingnetwork_dense_5_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_19AssignVariableOp=assignvariableop_19_valuenetwork_encodingnetwork_dense_5_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_20AssignVariableOp/assignvariableop_20_valuenetwork_dense_6_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp-assignvariableop_21_valuenetwork_dense_6_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ò
l
,__inference_function_with_signature_92040638
unknown:	 
identity	¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference_<lambda>_92039967^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
ð
°
*__inference_polymorphic_action_fn_92040852
	step_type

reward
discount
observationG
8normalize_observations_normalize_readvariableop_resource:	O
@normalize_observations_normalize_truediv_readvariableop_resource:	Y
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resource:	E
1sequential_2_dense_matmul_readvariableop_resource:
A
2sequential_2_dense_biasadd_readvariableop_resource:	G
3sequential_2_dense_1_matmul_readvariableop_resource:
C
4sequential_2_dense_1_biasadd_readvariableop_resource:	F
3sequential_2_dense_2_matmul_readvariableop_resource:	@B
4sequential_2_dense_2_biasadd_readvariableop_resource:@E
3sequential_2_dense_3_matmul_readvariableop_resource:@ B
4sequential_2_dense_3_biasadd_readvariableop_resource: T
Bsequential_2_means_projection_layer_matmul_readvariableop_resource: Q
Csequential_2_means_projection_layer_biasadd_readvariableop_resource:[
Msequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource:
identity¢/normalize_observations/normalize/ReadVariableOp¢Anormalize_observations/normalize/normalized_tensor/ReadVariableOp¢7normalize_observations/normalize/truediv/ReadVariableOp¢)sequential_2/dense/BiasAdd/ReadVariableOp¢(sequential_2/dense/MatMul/ReadVariableOp¢+sequential_2/dense_1/BiasAdd/ReadVariableOp¢*sequential_2/dense_1/MatMul/ReadVariableOp¢+sequential_2/dense_2/BiasAdd/ReadVariableOp¢*sequential_2/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard¢:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp¢9sequential_2/means_projection_layer/MatMul/ReadVariableOp¢Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp¥
/normalize_observations/normalize/ReadVariableOpReadVariableOp8normalize_observations_normalize_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7normalize_observations/normalize/truediv/ReadVariableOpReadVariableOp@normalize_observations_normalize_truediv_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
(normalize_observations/normalize/truedivRealDiv7normalize_observations/normalize/ReadVariableOp:value:0?normalize_observations/normalize/truediv/ReadVariableOp:value:0*
T0*
_output_shapes	
:}
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ö
6normalize_observations/normalize/normalized_tensor/addAddV2,normalize_observations/normalize/truediv:z:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes	
:£
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes	
:»
6normalize_observations/normalize/normalized_tensor/mulMulobservation<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes	
:*
dtype0®
6normalize_observations/normalize/normalized_tensor/NegNegInormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes	
:î
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *   À
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_2/dense/MatMul/ReadVariableOpReadVariableOp1sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0È
sequential_2/dense/MatMulMatMul>normalize_observations/normalize/clipped_normalized_tensor:z:00sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential_2/dense/BiasAddBiasAdd#sequential_2/dense/MatMul:product:01sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential_2/dense/TanhTanh#sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
sequential_2/dense_1/MatMulMatMulsequential_2/dense/Tanh:y:02sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_2/dense_1/BiasAddBiasAdd%sequential_2/dense_1/MatMul:product:03sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_2/dense_1/TanhTanh%sequential_2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0ª
sequential_2/dense_2/MatMulMatMulsequential_2/dense_1/Tanh:y:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_2/TanhTanh%sequential_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0ª
sequential_2/dense_3/MatMulMatMulsequential_2/dense_2/Tanh:y:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
sequential_2/dense_3/TanhTanh%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
9sequential_2/means_projection_layer/MatMul/ReadVariableOpReadVariableOpBsequential_2_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0È
*sequential_2/means_projection_layer/MatMulMatMulsequential_2/dense_3/Tanh:y:0Asequential_2/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpCsequential_2_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0â
+sequential_2/means_projection_layer/BiasAddBiasAdd4sequential_2/means_projection_layer/MatMul:product:0Bsequential_2/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda/zeros_like	ZerosLike4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpReadVariableOpMsequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ä
5sequential_2/nest_map/sequential_1/bias_layer/BiasAddBiasAdd"sequential_2/lambda/zeros_like:y:0Lsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/TanhTanh4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/mulMul$sequential_2/lambda_2/mul/x:output:0sequential_2/lambda_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/addAddV2$sequential_2/lambda_2/add/x:output:0sequential_2/lambda_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/SoftplusSoftplus>sequential_2/nest_map/sequential_1/bias_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :
Rsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:ï
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:ô
 sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿí
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: í
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0©sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskù
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack£sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:Û
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¸
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0¥sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0¡sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ö
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿÍ
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:¤
Zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¯
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¦
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:²
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceUsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0csequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskæ
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgssequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0]sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/AbsAbs,sequential_2/lambda_2/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
¥sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/LessLess£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
¦sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¢
¤sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/AllAll©sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Less:z:0¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Const:output:0*
_output_shapes
: ¢
­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = 
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_3Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = ²
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuardIf­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ø
else_branchÈRÅ
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92040751*
output_shapes
: *×
then_branchÇRÄ
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92040750©
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentity¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: ²
Tsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/group_depsNoOp½^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity*
_output_shapes
 }
8sequential_2/lambda_2/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2sequential_2/lambda_2/MultivariateNormalDiag/zerosFillYsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0Asequential_2/lambda_2/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
1sequential_2/lambda_2/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?s
1sequential_2/lambda_2/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ShapeShape;sequential_2/lambda_2/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:í
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
¸MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_sliceStridedSlice³MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Shape:output:0ÁMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskø
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB ï
¬MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1StridedSlice½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensor:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskù
µMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB û
·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB á
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsÀMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0»MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:Ü
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgs·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs:r0:0½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:
°MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shapeIdentity¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:Ó
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : é
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_sliceStridedSlice¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shape:output:0§MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskß
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB á
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgsBroadcastArgs¦MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1:output:0¡MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:è
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:½
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:²
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :È
~MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskÎ
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1Shape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:´
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSlice{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask¾
{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB À
}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgsMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:¬
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgs}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:¨
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:º
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:¢
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shapeIdentityyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ú
GMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concatConcatV2MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shape:output:0UMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:þ
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastToBroadcastTosequential_2/lambda_2/add:z:0PMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Deterministic/sample/ShapeShapeUMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastTo:output:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:â
 Deterministic/sample/BroadcastToBroadcastToUMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastTo:output:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¬
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
NoOpNoOp0^normalize_observations/normalize/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOp8^normalize_observations/normalize/truediv/ReadVariableOp*^sequential_2/dense/BiasAdd/ReadVariableOp)^sequential_2/dense/MatMul/ReadVariableOp,^sequential_2/dense_1/BiasAdd/ReadVariableOp+^sequential_2/dense_1/MatMul/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp´^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard;^sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:^sequential_2/means_projection_layer/MatMul/ReadVariableOpE^sequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/normalize_observations/normalize/ReadVariableOp/normalize_observations/normalize/ReadVariableOp2
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2r
7normalize_observations/normalize/truediv/ReadVariableOp7normalize_observations/normalize/truediv/ReadVariableOp2V
)sequential_2/dense/BiasAdd/ReadVariableOp)sequential_2/dense/BiasAdd/ReadVariableOp2T
(sequential_2/dense/MatMul/ReadVariableOp(sequential_2/dense/MatMul/ReadVariableOp2Z
+sequential_2/dense_1/BiasAdd/ReadVariableOp+sequential_2/dense_1/BiasAdd/ReadVariableOp2X
*sequential_2/dense_1/MatMul/ReadVariableOp*sequential_2/dense_1/MatMul/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2ì
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard2x
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp2v
9sequential_2/means_projection_layer/MatMul/ReadVariableOp9sequential_2/means_projection_layer/MatMul/ReadVariableOp2
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpDsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	step_type:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namereward:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
discount:UQ
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameobservation
Û
f
&__inference_signature_wrapper_92040646
unknown:	 
identity	¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_92040638^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
Ç
>
,__inference_function_with_signature_92040626

batch_sizeÿ
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_get_initial_state_92040625*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ð
¸
*__inference_polymorphic_action_fn_92040550
	time_step
time_step_1
time_step_2
time_step_3G
8normalize_observations_normalize_readvariableop_resource:	O
@normalize_observations_normalize_truediv_readvariableop_resource:	Y
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resource:	E
1sequential_2_dense_matmul_readvariableop_resource:
A
2sequential_2_dense_biasadd_readvariableop_resource:	G
3sequential_2_dense_1_matmul_readvariableop_resource:
C
4sequential_2_dense_1_biasadd_readvariableop_resource:	F
3sequential_2_dense_2_matmul_readvariableop_resource:	@B
4sequential_2_dense_2_biasadd_readvariableop_resource:@E
3sequential_2_dense_3_matmul_readvariableop_resource:@ B
4sequential_2_dense_3_biasadd_readvariableop_resource: T
Bsequential_2_means_projection_layer_matmul_readvariableop_resource: Q
Csequential_2_means_projection_layer_biasadd_readvariableop_resource:[
Msequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource:
identity¢/normalize_observations/normalize/ReadVariableOp¢Anormalize_observations/normalize/normalized_tensor/ReadVariableOp¢7normalize_observations/normalize/truediv/ReadVariableOp¢)sequential_2/dense/BiasAdd/ReadVariableOp¢(sequential_2/dense/MatMul/ReadVariableOp¢+sequential_2/dense_1/BiasAdd/ReadVariableOp¢*sequential_2/dense_1/MatMul/ReadVariableOp¢+sequential_2/dense_2/BiasAdd/ReadVariableOp¢*sequential_2/dense_2/MatMul/ReadVariableOp¢+sequential_2/dense_3/BiasAdd/ReadVariableOp¢*sequential_2/dense_3/MatMul/ReadVariableOp¢³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard¢:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp¢9sequential_2/means_projection_layer/MatMul/ReadVariableOp¢Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp¥
/normalize_observations/normalize/ReadVariableOpReadVariableOp8normalize_observations_normalize_readvariableop_resource*
_output_shapes	
:*
dtype0µ
7normalize_observations/normalize/truediv/ReadVariableOpReadVariableOp@normalize_observations_normalize_truediv_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
(normalize_observations/normalize/truedivRealDiv7normalize_observations/normalize/ReadVariableOp:value:0?normalize_observations/normalize/truediv/ReadVariableOp:value:0*
T0*
_output_shapes	
:}
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ö
6normalize_observations/normalize/normalized_tensor/addAddV2,normalize_observations/normalize/truediv:z:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes	
:£
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes	
:»
6normalize_observations/normalize/normalized_tensor/mulMultime_step_3<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes	
:*
dtype0®
6normalize_observations/normalize/normalized_tensor/NegNegInormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0*
T0*
_output_shapes	
:ß
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes	
:î
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *   À
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_2/dense/MatMul/ReadVariableOpReadVariableOp1sequential_2_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0È
sequential_2/dense/MatMulMatMul>normalize_observations/normalize/clipped_normalized_tensor:z:00sequential_2/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_2/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_2_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential_2/dense/BiasAddBiasAdd#sequential_2/dense/MatMul:product:01sequential_2/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential_2/dense/TanhTanh#sequential_2/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
*sequential_2/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
sequential_2/dense_1/MatMulMatMulsequential_2/dense/Tanh:y:02sequential_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¶
sequential_2/dense_1/BiasAddBiasAdd%sequential_2/dense_1/MatMul:product:03sequential_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sequential_2/dense_1/TanhTanh%sequential_2/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0ª
sequential_2/dense_2/MatMulMatMulsequential_2/dense_1/Tanh:y:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
sequential_2/dense_2/TanhTanh%sequential_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0ª
sequential_2/dense_3/MatMulMatMulsequential_2/dense_2/Tanh:y:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0µ
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
sequential_2/dense_3/TanhTanh%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
9sequential_2/means_projection_layer/MatMul/ReadVariableOpReadVariableOpBsequential_2_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0È
*sequential_2/means_projection_layer/MatMulMatMulsequential_2/dense_3/Tanh:y:0Asequential_2/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpCsequential_2_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0â
+sequential_2/means_projection_layer/BiasAddBiasAdd4sequential_2/means_projection_layer/MatMul:product:0Bsequential_2/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda/zeros_like	ZerosLike4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpReadVariableOpMsequential_2_nest_map_sequential_1_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ä
5sequential_2/nest_map/sequential_1/bias_layer/BiasAddBiasAdd"sequential_2/lambda/zeros_like:y:0Lsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/TanhTanh4sequential_2/means_projection_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/mulMul$sequential_2/lambda_2/mul/x:output:0sequential_2/lambda_2/Tanh:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
sequential_2/lambda_2/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sequential_2/lambda_2/addAddV2$sequential_2/lambda_2/add/x:output:0sequential_2/lambda_2/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2/lambda_2/SoftplusSoftplus>sequential_2/nest_map/sequential_1/bias_layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :
Rsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:ï
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:ô
 sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿí
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: í
¢sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0©sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0«sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskù
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack£sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:Û
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¸
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0¥sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0¡sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ö
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿÍ
sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlicesequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Lsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:¤
Zsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¯
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¦
\sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:²
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceUsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0csequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0esequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskæ
Tsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgssequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0]sequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/AbsAbs,sequential_2/lambda_2/Softplus:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
¥sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/LessLess£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿø
¦sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¢
¤sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/AllAll©sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Less:z:0¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Const:output:0*
_output_shapes
: ¢
­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = 
¯sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/Const_3Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = ²
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuardIf­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0­sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/All:output:0£sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:output:0sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:y:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ø
else_branchÈRÅ
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92040449*
output_shapes
: *×
then_branchÇRÄ
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92040448©
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentity¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: ²
Tsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/group_depsNoOp½^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity*
_output_shapes
 }
8sequential_2/lambda_2/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
2sequential_2/lambda_2/MultivariateNormalDiag/zerosFillYsequential_2/lambda_2/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0Asequential_2/lambda_2/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
1sequential_2/lambda_2/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?s
1sequential_2/lambda_2/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ShapeShape;sequential_2/lambda_2/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:í
ªMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
¸MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_sliceStridedSlice³MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Shape:output:0ÁMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_1:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskø
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB ï
¬MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B : 
ºMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
¼MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1StridedSlice½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/shape_as_tensor:output:0ÃMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_1:output:0ÅMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskù
µMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB û
·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB á
²MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgsBroadcastArgsÀMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs/s0_1:output:0»MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:Ü
´MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1BroadcastArgs·MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs:r0:0½MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:
°MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shapeIdentity¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:Ó
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B : é
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ë
 MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_sliceStridedSlice¹MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/batch_shape:output:0§MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_1:output:0©MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskß
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB á
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgsBroadcastArgs¦MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs/s0_1:output:0¡MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:è
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:½
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ShapeShapesequential_2/lambda_2/add:z:0*
T0*
_output_shapes
:²
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :È
~MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_sliceStridedSliceyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskÎ
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1Shape,sequential_2/lambda_2/Softplus:activations:0*
T0*
_output_shapes
:´
rMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Const_1Const*
_output_shapes
: *
dtype0*
value	B :Ë
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Í
MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1StridedSlice{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/Shape_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask¾
{MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB À
}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ²
xMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgsBroadcastArgsMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs/s0_1:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice:output:0*
_output_shapes
:¬
zMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1BroadcastArgs}MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs:r0:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/strided_slice_1:output:0*
_output_shapes
:¨
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentityMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/BroadcastArgs_1:r0:0*
T0*
_output_shapes
:º
pMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:¢
vMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shapeIdentityyMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ú
GMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concatConcatV2MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0MultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/MultivariateNormalDiag/event_shape_tensor/event_shape:output:0UMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:þ
LMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastToBroadcastTosequential_2/lambda_2/add:z:0PMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 
Deterministic/sample/ShapeShapeUMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastTo:output:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:â
 Deterministic/sample/BroadcastToBroadcastToUMultivariateNormalDiag_CONSTRUCTED_AT_sequential_2_lambda_2/mode/BroadcastTo:output:0$Deterministic/sample/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¬
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityclip_by_value:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
NoOpNoOp0^normalize_observations/normalize/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOp8^normalize_observations/normalize/truediv/ReadVariableOp*^sequential_2/dense/BiasAdd/ReadVariableOp)^sequential_2/dense/MatMul/ReadVariableOp,^sequential_2/dense_1/BiasAdd/ReadVariableOp+^sequential_2/dense_1/MatMul/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp´^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard;^sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:^sequential_2/means_projection_layer/MatMul/ReadVariableOpE^sequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2b
/normalize_observations/normalize/ReadVariableOp/normalize_observations/normalize/ReadVariableOp2
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2r
7normalize_observations/normalize/truediv/ReadVariableOp7normalize_observations/normalize/truediv/ReadVariableOp2V
)sequential_2/dense/BiasAdd/ReadVariableOp)sequential_2/dense/BiasAdd/ReadVariableOp2T
(sequential_2/dense/MatMul/ReadVariableOp(sequential_2/dense/MatMul/ReadVariableOp2Z
+sequential_2/dense_1/BiasAdd/ReadVariableOp+sequential_2/dense_1/BiasAdd/ReadVariableOp2X
*sequential_2/dense_1/MatMul/ReadVariableOp*sequential_2/dense_1/MatMul/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2ì
³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard³sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard2x
:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp:sequential_2/means_projection_layer/BiasAdd/ReadVariableOp2v
9sequential_2/means_projection_layer/MatMul/ReadVariableOp9sequential_2/means_projection_layer/MatMul/ReadVariableOp2
Dsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOpDsequential_2/nest_map/sequential_1/bias_layer/BiasAdd/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:SO
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step
ó
	
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92040949æ
ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ä
¿sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholderÆ
Ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholder_1Ã
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
×
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ´
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ê>
ë
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92041149ä
ßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ú
Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constØ
Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_absÃ
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
¢ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert¶
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.­
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:ª
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = ¨
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = °
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/AssertAssertßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_allÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2:output:0Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4:output:0Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_abs*
T

2*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ð
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¹
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ2ú
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assertºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
×
.
,__inference_function_with_signature_92040649ç
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference_<lambda>_92039970*(
_construction_contextkEagerRuntime*
_input_shapes 
^

__inference_<lambda>_92039970*(
_construction_contextkEagerRuntime*
_input_shapes 
¿
8
&__inference_get_initial_state_92040625

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
¢

&__inference_signature_wrapper_92040619
discount
observation

reward
	step_type
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	
	unknown_6:	@
	unknown_7:@
	unknown_8:@ 
	unknown_9: 

unknown_10: 

unknown_11:

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_92040581o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
0/discount:WS
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_name0/observation:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_name0/step_type
à
(
&__inference_signature_wrapper_92040653ö
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_92040649*(
_construction_contextkEagerRuntime*
_input_shapes 
ó
	
Ásequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_true_92040750æ
ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ä
¿sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholderÆ
Ásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_placeholder_1Ã
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
×
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityásequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ´
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ê>
ë
Âsequential_2_lambda_2_MultivariateNormalDiag_scale_matvec_linear_operator_LinearOperatorDiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_Assert_AssertGuard_false_92040449ä
ßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all
Ú
Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constØ
Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_absÃ
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1
¢ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert¶
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*D
value;B9 B3Singular operator:  Diagonal contained zero values.­
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:ª
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*·
value­Bª B£x (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Const:0) = ¨
Ásequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*µ
value«B¨ B¡y (sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/Abs:0) = °
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/AssertAssertßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_allÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_0:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_1:output:0Êsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_2:output:0Õsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_constÊsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert/data_4:output:0Ósequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_abs*
T

2*
_output_shapes
 
¼sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/IdentityIdentityßsequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_assert_sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_all»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ð
¾sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1IdentityÅsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity:output:0¹^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¹
¸sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/NoOpNoOp»^sequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert*"
_acd_function_control_output(*
_output_shapes
 "
¾sequential_2_lambda_2_multivariatenormaldiag_scale_matvec_linear_operator_linearoperatordiag_assert_non_singular_assert_no_entries_with_modulus_zero_assert_less_assert_assertguard_identity_1Çsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
: : :ÿÿÿÿÿÿÿÿÿ2ú
ºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assertºsequential_2/lambda_2/MultivariateNormalDiag/scale_matvec_linear_operator/LinearOperatorDiag/assert_non_singular/assert_no_entries_with_modulus_zero/assert_less/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"ÛL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ä
action¹
4

0/discount&
action_0_discount:0ÿÿÿÿÿÿÿÿÿ
?
0/observation.
action_0_observation:0ÿÿÿÿÿÿÿÿÿ
0
0/reward$
action_0_reward:0ÿÿÿÿÿÿÿÿÿ
6
0/step_type'
action_0_step_type:0ÿÿÿÿÿÿÿÿÿ:
action0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:ç
Í

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
¿
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20"
trackable_tuple_wrapper
5
 _wrapped_policy"
trackable_dict_wrapper
2
*__inference_polymorphic_action_fn_92040852
*__inference_polymorphic_action_fn_92041051±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
0__inference_polymorphic_distribution_fn_92041225±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
&__inference_get_initial_state_92041228¦
²
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
__inference_<lambda>_92039970"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
__inference_<lambda>_92039967"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
`

!action
"get_initial_state
#get_train_step
$get_metadata"
signature_map
-:+
2sequential_2/dense/kernel
&:$2sequential_2/dense/bias
/:-
2sequential_2/dense_1/kernel
(:&2sequential_2/dense_1/bias
.:,	@2sequential_2/dense_2/kernel
':%@2sequential_2/dense_2/bias
-:+@ 2sequential_2/dense_3/kernel
':% 2sequential_2/dense_3/bias
<:: 2*sequential_2/means_projection_layer/kernel
6:42(sequential_2/means_projection_layer/bias
@:>22sequential_2/nest_map/sequential_1/bias_layer/bias
:2avg_0
:2count_0
:2m2_0
:2
m2_carry_0
>:<	K2+ValueNetwork/EncodingNetwork/dense_4/kernel
7:5K2)ValueNetwork/EncodingNetwork/dense_4/bias
=:;K(2+ValueNetwork/EncodingNetwork/dense_5/kernel
7:5(2)ValueNetwork/EncodingNetwork/dense_5/bias
-:+(2ValueNetwork/dense_6/kernel
':%2ValueNetwork/dense_6/bias
c
%_actor_network
&_observation_normalizer
'_value_network"
_generic_user_object
ôBñ
&__inference_signature_wrapper_92040619
0/discount0/observation0/reward0/step_type"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÐBÍ
&__inference_signature_wrapper_92040631
batch_size"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÂB¿
&__inference_signature_wrapper_92040646"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÂB¿
&__inference_signature_wrapper_92040653"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í
(_layer_state_is_list
)_sequential_layers
*_layer_has_state
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
e
1_flat_variable_spec

2_count
3_avg
4_m2
5	_m2_carry"
_generic_user_object
Ï
6_encoder
7_postprocessing_layers
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
Á
K_postprocessing_layers
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
æ2ãà
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ãà
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
»

kernel
bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
¦
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ò
_state_spec
_nested_layers
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
0
1
2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
æ2ãà
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ãà
×²Ó
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
¢ 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
´
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_dict_wrapper
6
ºloc

»scale"
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
«
Æ	variables
Çtrainable_variables
Èregularization_losses
É	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

kernel
bias
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

kernel
bias
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
ö
Ø_layer_state_is_list
Ù_sequential_layers
Ú_layer_has_state
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
ö
á_layer_state_is_list
â_sequential_layers
ã_layer_has_state
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
0
º0
»1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
Æ	variables
Çtrainable_variables
Èregularization_losses
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
(
ù0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
 "
trackable_list_wrapper
(
ÿ0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
â2ßÜ
Ó²Ï
FullArgSpec.
args&#
jself
jinputs
jnetwork_state
varargs
 
varkwjkwargs
defaults
¢ 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
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
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
(
ù0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
µ
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
(
ÿ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2ÃÀ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper<
__inference_<lambda>_92039967¢

¢ 
ª " 	5
__inference_<lambda>_92039970¢

¢ 
ª "ª S
&__inference_get_initial_state_92041228)"¢
¢


batch_size 
ª "¢ ù
*__inference_polymorphic_action_fn_92040852Êß¢Û
Ó¢Ï
Ç²Ã
TimeStep,
	step_type
	step_typeÿÿÿÿÿÿÿÿÿ&
reward
rewardÿÿÿÿÿÿÿÿÿ*
discount
discountÿÿÿÿÿÿÿÿÿ5
observation&#
observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "V²S

PolicyStep*
action 
actionÿÿÿÿÿÿÿÿÿ
state¢ 
info¢ ¡
*__inference_polymorphic_action_fn_92041051ò¢
û¢÷
ï²ë
TimeStep6
	step_type)&
time_step/step_typeÿÿÿÿÿÿÿÿÿ0
reward&#
time_step/rewardÿÿÿÿÿÿÿÿÿ4
discount(%
time_step/discountÿÿÿÿÿÿÿÿÿ?
observation0-
time_step/observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "V²S

PolicyStep*
action 
actionÿÿÿÿÿÿÿÿÿ
state¢ 
info¢ ö
0__inference_polymorphic_distribution_fn_92041225Áß¢Û
Ó¢Ï
Ç²Ã
TimeStep,
	step_type
	step_typeÿÿÿÿÿÿÿÿÿ&
reward
rewardÿÿÿÿÿÿÿÿÿ*
discount
discountÿÿÿÿÿÿÿÿÿ5
observation&#
observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "Ì²È

PolicyStep
actionÁ¢½
`
FªC

atol 

locÿÿÿÿÿÿÿÿÿ

rtol 
JªG

allow_nan_statsp

namejDeterministic_1

validate_argsp 
¢
j
parameters
¢ 
¢
jnameEtf_agents.policies.greedy_policy.DeterministicWithLogProb_ACTTypeSpec 
state¢ 
info¢ È
&__inference_signature_wrapper_92040619Ù¢Õ
¢ 
ÍªÉ
.

0/discount 

0/discountÿÿÿÿÿÿÿÿÿ
9
0/observation(%
0/observationÿÿÿÿÿÿÿÿÿ
*
0/reward
0/rewardÿÿÿÿÿÿÿÿÿ
0
0/step_type!
0/step_typeÿÿÿÿÿÿÿÿÿ"/ª,
*
action 
actionÿÿÿÿÿÿÿÿÿa
&__inference_signature_wrapper_9204063170¢-
¢ 
&ª#
!

batch_size

batch_size "ª Z
&__inference_signature_wrapper_920406460¢

¢ 
ª "ª

int64
int64 	>
&__inference_signature_wrapper_92040653¢

¢ 
ª "ª 