use crate::save::Save;
use std::{
	io::{Error as IOError,Read as IORead,Write as IOWrite},
	time::{SystemTime,UNIX_EPOCH}
};
#[derive(Clone,Debug)]
pub struct AIBlock{buffer:Vec<f32>,bufferderivatives:Vec<f32>,bufferlen:usize,bufferoffset:usize,derivativesoffset:usize,pub parameters:Vec<f32>,parametersderivatives:Vec<f32>,parametersposition:usize,pub seed:u128}
#[derive(Clone,Debug)]
pub struct GPT{block:AIBlock,contextdimension:usize,headcount:usize,headdimension:usize,transformerlayers:usize,uniquetokencount:usize}
#[derive(Clone,Debug)]
pub struct GeluMultilayerPerceptron{block:AIBlock,inputdimension:usize,intermediatedimension:usize,intermediatelayers:usize,outputdimension:usize}
#[derive(Clone,Debug)]//TODO think of a better name for this structure since this acronym is meaningless. Perhaps something about testing because testing attention since that is the entire point of writing this
pub struct SLT{block:AIBlock,pub contextdimension:usize,pub embeddingdimension:usize,pub heads:usize,pub intermediatedimension:usize,pub outputdimension:usize}
fn allocate_data<'a>(compact:bool,data:&'a mut Vec<f32>,datalen:usize,newlen:usize,offset:&mut usize)->(&'a mut [f32],&'a mut [f32]){
	let datastart=*offset;
	if compact&&datastart>=newlen{
		*offset=0;
		let (new,data)=data.split_at_mut(datastart);
		(&mut data[..datalen],&mut new[..newlen])
	}else{
		let newstart=datalen+datastart;
		*offset=newstart;
		let newstop=newlen+newstart;
		if data.len()<newstop{data.resize(newstop,0.0)}
		let (data,new)=data.split_at_mut(newstart);
		(&mut data[datastart..],&mut new[..newlen])
	}
}fn allocate_parameters<'a>(len:usize,parameters:&'a mut Vec<f32>,position:&mut usize,seed:&mut u128)->&'a [f32]{
	let start=*position;
	let stop=len+start;
	*position=stop;
	if parameters.len()<stop{parameters.resize_with(stop,||rfloat(seed));}
	&parameters[start..stop]
}fn reverse_data<'a>(data:&'a mut Vec<f32>,datalen:usize,offset:&mut usize,oldlen:usize)->(&'a mut [f32],&'a mut [f32]){
	let datastart=*offset;
	let oldstart=datastart-oldlen;
	*offset=oldstart;
	let (old,data)=data.split_at_mut(datastart);
	(&mut old[oldstart..],&mut data[..datalen])
}fn reverse_parameters_and_parameters_derivatives<'a>(len:usize,parameters:&'a Vec<f32>,parametersderivatives:&'a mut Vec<f32>,position:&mut usize)->(&'a [f32],&'a mut [f32]){
	let stop=*position;
	if parametersderivatives.len()<stop{parametersderivatives.resize(stop,0.0)}
	let start=stop-len;
	*position=start;
	(&parameters[start..stop],&mut parametersderivatives[start..stop])
}impl AIBlock{
	fn from_existing_data(buffer:Vec<f32>,bufferderivatives:Vec<f32>,bufferlen:usize,bufferoffset:usize,derivativesoffset:usize,parameters:Vec<f32>,parametersderivatives:Vec<f32>,parametersposition:usize,seed:u128)->Self{
		Self{buffer,bufferderivatives,bufferlen,bufferoffset,derivativesoffset,parameters,parametersderivatives,parametersposition,seed}
	}pub fn allocate(&mut self,compact:bool,differentiate:bool,layerlen:usize,parameterslen:usize,templen:usize)->(&[f32],&mut [f32],&mut [f32],&[f32],&[f32],&mut [f32],&mut [f32],&mut [f32]){//TODO unbackwards the buffer derivatives in noncompact differentiate mode
		let (buffer,bufferderivatives,bufferlen,bufferoffset,derivativesoffset,parameters,parametersderivatives,parametersposition,seed)=(&mut self.buffer,&mut self.bufferderivatives,self.bufferlen,&mut self.bufferoffset,&mut self.derivativesoffset,&mut self.parameters,&mut self.parametersderivatives,&mut self.parametersposition,&mut self.seed);
		self.bufferlen=layerlen;
		if differentiate{
			let ((input,output),(outputderivatives,inputderivatives),(parameters,parametersderivatives))=(reverse_data(buffer,bufferlen,bufferoffset,layerlen+templen),allocate_data(compact,bufferderivatives,bufferlen,layerlen+templen,derivativesoffset),reverse_parameters_and_parameters_derivatives(parameterslen,parameters,parametersderivatives,parametersposition));
			let ((input,temp),(inputderivatives,tempderivatives))=(input.split_at_mut(layerlen),inputderivatives.split_at_mut(layerlen));
			(input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)
		}else{
			let ((input,output),(outputderivatives,inputderivatives),(parameters,parametersderivatives))=(allocate_data(compact,buffer,bufferlen,layerlen+templen,bufferoffset),bufferderivatives[..0].split_at_mut(0),(allocate_parameters(parameterslen,parameters,parametersposition,seed),&mut parametersderivatives[..0]));
			let ((temp,output),(inputderivatives,tempderivatives))=(output.split_at_mut(templen),inputderivatives.split_at_mut(0));
			self.bufferoffset+=templen;
			(input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)
		}
	}pub fn adjust(&mut self,persistence:f32,step:f32){
		self.parameters.iter_mut().zip(&mut self.parametersderivatives).for_each(|(x,dx)|{
			*x+=*dx*step;
			*dx*=persistence;
		});
	}pub fn affine_layer(&mut self,compact:bool,differentiate:bool,inputdimension:usize,outputdimension:usize){
		let bufferlen=self.bufferlen;
		let layerlen=if differentiate{bufferlen/outputdimension*inputdimension}else{bufferlen/inputdimension*outputdimension};
		let (input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,_,_)=self.allocate(compact,differentiate,layerlen,(inputdimension+1)*outputdimension,0);
		if differentiate{
			inputderivatives.fill(0.0);
			rd_trans_mat_mul(inputderivatives,parametersderivatives,outputderivatives,BIAS,input,inputdimension,parameters);
		}else{
			trans_mat_mul(BIAS,input,inputdimension,parameters,output);
		}
	}pub fn attention_layer(&mut self,compact:bool,contextdimension:usize,differentiate:bool,headcount:usize,headdimension:usize){//TODO make this better written compact version and dont necessarily allocate entire context requirement for compact inference
		use std::ops::Range;
		fn make_range(len:usize,start:usize)->Range<usize>{start..len+start}
		fn two_subslices(r0:Range<usize>,r1:Range<usize>,slice:&mut [f32])->(&mut [f32],&mut [f32]){
			fn make_subslices(higher:Range<usize>,lower:Range<usize>,slice:&mut [f32])->(&mut [f32],&mut [f32]){
				let (a,b)=slice.split_at_mut(higher.start);
				(&mut a[lower.start..lower.end],&mut b[..higher.len()])
			}if r0.start>=r1.end{make_subslices(r0,r1,slice)}else{make_subslices(r1,r0,slice)}
		}let (attention_volume,sharpness)=(|n:usize|headcount*(n+1)*n/2,(headdimension as f32).powf(-0.5));
		let (attention_range,kqv_range)=(|contextdimension:usize,n:usize|make_range(headcount*(n+1),attention_volume(n)),|head:usize,n:usize|make_range(headdimension,head*headdimension+headcount*headdimension*n));

		let (attentionarea,embeddingarea,len)=((contextdimension+1)*contextdimension/2,headcount*headdimension,self.bufferlen);
		let (attentionvolume,contextvolume,kqvheadparametervolume,kqvparametercount)=(attentionarea*headcount,contextdimension*embeddingarea,embeddingarea*headdimension,embeddingarea*embeddingarea);
		let extradimension=len%contextvolume/embeddingarea;
		let attentionrequirement=len/contextdimension*attentionvolume+(extradimension+1)*extradimension/2*headcount;
		let (input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)=self.allocate(compact,differentiate,len,embeddingarea+kqvparametercount*4,attentionrequirement+len*4);
		let ((attention,temp),(keyparameters,parameters))=(temp.split_at_mut(attentionrequirement),parameters.split_at(kqvparametercount));
		let ((keys,temp),(queryparameters,parameters))=(temp.split_at_mut(len),parameters.split_at(kqvparametercount));
		let ((queries,temp),(valueparameters,outputparameters))=(temp.split_at_mut(len),parameters.split_at(kqvparametercount));
		let (values,summaries)=temp.split_at_mut(len);
		if differentiate{
			inputderivatives.copy_from_slice(outputderivatives);
			let ((attentionderivatives,tempderivatives),(keyparametersderivatives,parametersderivatives))=(tempderivatives.split_at_mut(attentionrequirement),parametersderivatives.split_at_mut(kqvparametercount));
			tempderivatives.fill(0.0);
			let ((keysderivatives,tempderivatives),(queryparametersderivatives,parametersderivatives))=(tempderivatives.split_at_mut(len),parametersderivatives.split_at_mut(kqvparametercount));
			let ((queriesderivatives,tempderivatives),(valueparametersderivatives,outputparametersderivatives))=(tempderivatives.split_at_mut(len),parametersderivatives.split_at_mut(kqvparametercount));
			let (valuesderivatives,summariesderivatives)=tempderivatives.split_at_mut(len);
			rd_trans_mat_mul(summariesderivatives,outputparametersderivatives,outputderivatives,ACC|BIAS,summaries,embeddingarea,outputparameters);
			attention.chunks(attentionvolume).zip(attentionderivatives.chunks_mut(attentionvolume)).zip(keys.chunks(contextvolume)).zip(keysderivatives.chunks_mut(contextvolume)).zip(queries.chunks(contextvolume)).zip(queriesderivatives.chunks_mut(contextvolume)).zip(values.chunks(contextvolume)).zip(valuesderivatives.chunks_mut(contextvolume)).zip(summariesderivatives.chunks(contextvolume)).for_each(|((((((((attention,attentionderivatives),keys),keysderivatives),queries),queriesderivatives),values),valuesderivatives),summariesderivatives)|{
				let contextdimension=summariesderivatives.len()/embeddingarea;
				summariesderivatives.chunks_exact(embeddingarea).enumerate().for_each(|(n,summaryderivatives)|{
					let (attentiondimension,attentionrange)=(n+1,attention_range(contextdimension,n));
					let (attention,attentionderivatives)=(&attention[attentionrange.clone()],&mut attentionderivatives[attentionrange.clone()]);
					attention.chunks_exact(attentiondimension).enumerate().zip(attentionderivatives.chunks_exact_mut(attentiondimension)).zip(summaryderivatives.chunks_exact(headdimension)).for_each(|(((h,attention),attentionderivatives),summaryvaluederivatives)|{
						let queryrange=kqv_range(h,n);
						let query=&queries[queryrange.clone()];
						let sum=attention.iter().enumerate().zip(&mut *attentionderivatives).map(|((v,&a),da)|{
							let valuerange=kqv_range(h,v);
							let (value,valuederivatives)=(&values[valuerange.clone()],&mut valuesderivatives[valuerange.clone()]);
							*da=summaryvaluederivatives.iter().zip(value).zip(valuederivatives).map(|((ds,v),dv)|{
								*dv+=a*ds;
								ds*v
							}).sum();
							*da*a
						}).sum::<f32>();
						attention.iter().enumerate().zip(&*attentionderivatives).for_each(|((k,a),da)|{
							let keyrange=kqv_range(h,k);
							let (key,keyderivatives,query,queryderivatives)=(&keys[keyrange.clone()],&mut keysderivatives[keyrange.clone()],&queries[keyrange.clone()],&mut queriesderivatives[keyrange.clone()]);
							rd_dot(key,query,keyderivatives,queryderivatives,(da-sum)*a);
						});
					});
				});
			});
			rd_trans_mat_mul(inputderivatives,keyparametersderivatives,keysderivatives,0,input,embeddingarea,keyparameters);
			rd_trans_mat_mul(inputderivatives,queryparametersderivatives,queriesderivatives,0,input,embeddingarea,queryparameters);
			rd_trans_mat_mul(inputderivatives,valueparametersderivatives,valuesderivatives,0,input,embeddingarea,valueparameters);
		}else{
			output.copy_from_slice(input);
			summaries.fill(0.0);
			trans_mat_mul(0,input,embeddingarea,keyparameters,keys);
			trans_mat_mul(0,input,embeddingarea,queryparameters,queries);
			trans_mat_mul(0,input,embeddingarea,valueparameters,values);
			attention.chunks_mut(attentionvolume).zip(keys.chunks(contextvolume)).zip(queries.chunks(contextvolume)).zip(values.chunks(contextvolume)).zip(summaries.chunks_mut(contextvolume)).for_each(|((((attention,keys),queries),values),summaries)|{
				let contextdimension=summaries.len()/embeddingarea;
				summaries.chunks_exact_mut(embeddingarea).enumerate().for_each(|(n,summary)|{
					let (attention,attentiondimension)=(&mut attention[attention_range(contextdimension,n)],n+1);
					attention.chunks_exact_mut(attentiondimension).enumerate().zip(summary.chunks_exact_mut(headdimension)).for_each(|((h,attention),summaryvalue)|{
						let query=&queries[kqv_range(h,n)];
						let max=attention.iter_mut().enumerate().map(|(k,a)|{
							*a=dot(&keys[kqv_range(h,k)],query);
							*a
						}).fold(f32::NEG_INFINITY,f32::max);
						let scale=attention.iter_mut().map(|a|{
							*a=if *a==max{1.0}else{((*a-max)*sharpness).exp()};
							*a
						}).sum::<f32>().recip();
						attention.iter_mut().enumerate().for_each(|(v,a)|{
							*a*=scale;
							let (a,value)=(*a,&values[kqv_range(h,v)]);
							summaryvalue.iter_mut().zip(value).for_each(|(s,v)|*s+=a*v);
						});
					});
				});
			});
			trans_mat_mul(ACC|BIAS,summaries,embeddingarea,outputparameters,output);
		}
	}pub fn compactify(&mut self){(self.buffer,self.bufferderivatives,self.parametersderivatives)=Default::default()}
	pub fn differentiable_output(&mut self)->(&[f32],&mut [f32]){
		let (buffer,bufferderivatives,bufferlen,bufferoffset,derivativesstart)=(&self.buffer,&mut self.bufferderivatives,self.bufferlen,self.bufferoffset,self.derivativesoffset);
		let derivativesstop=bufferlen+derivativesstart;
		if bufferderivatives.len()<derivativesstop{bufferderivatives.resize(derivativesstop,0.0)}
		(&buffer[bufferoffset..bufferlen+bufferoffset],&mut bufferderivatives[derivativesstart..derivativesstop])
	}pub fn differentiate_entropic_one_hot_error<T:Copy+Into<usize>>(&mut self,correct:&[T],mut errorderivative:f32,dimension:usize,sharpness:f32){
		let (logits,logitsderivatives)=self.differentiable_output();
		errorderivative/=logits.len()as f32;
		correct.iter().zip(logits.chunks_exact(dimension)).zip(logitsderivatives.chunks_exact_mut(dimension)).for_each(|((&correct,logits),logitsderivatives)|rd_entropic_one_hot_error(correct.into(),errorderivative,logits,logitsderivatives,sharpness));
	}pub fn differentiate_squared_error(&mut self,errorderivative:f32,target:&[f32]){
		let (output,outputderivatives)=self.differentiable_output();
		rd_squared_error(errorderivative,output,outputderivatives,target);
	}pub fn embed<T:Copy+Into<usize>>(&mut self,compact:bool,contextdimension:usize,differentiate:bool,embeddingdimension:usize,priorlen:usize,tokens:&[T],uniquetokencount:usize){
		let (positionencodingarea,tokenencodingarea)=(contextdimension*embeddingdimension,embeddingdimension*uniquetokencount);
		let (_,_,output,outputderivatives,parameters,parametersderivatives,_,_)=self.allocate(compact,differentiate,if differentiate{priorlen}else{embeddingdimension*tokens.len()},positionencodingarea+tokenencodingarea,0);
		if differentiate{
			let (_,tokenencodingsderivatives)=parametersderivatives.split_at_mut(positionencodingarea);//TODO there should be a more efficient way to exclude position encodings from differentiation, and it should also be optional
			outputderivatives.chunks_exact(embeddingdimension).zip(tokens.iter().map(|&t|embeddingdimension*t.into())).for_each(|(outputderivatives,tx)|outputderivatives.iter().zip(&mut tokenencodingsderivatives[tx..embeddingdimension+tx]).for_each(|(d,dt)|*dt+=d));
		}else{
			let (positionencodings,tokenencodings)=parameters.split_at(positionencodingarea);
			output.chunks_exact_mut(embeddingdimension).zip(positionencodings.chunks_exact(embeddingdimension).cycle()).zip(tokens.iter().map(|&t|embeddingdimension*t.into())).for_each(|((output,positionencoding),tx)|output.iter_mut().zip(&tokenencodings[tx..embeddingdimension+tx]).zip(positionencoding).for_each(|((o,t),p)|*o=p+t));
		}
	}pub fn entropic_one_hot_error<T:Copy+Into<usize>>(&self,correct:&[T],dimension:usize,sharpness:f32)->f32{
		let logits=self.output().chunks_exact(dimension);
		let l=logits.len()as f32;
		correct.iter().zip(logits).map(|(&correct,logits)|entropic_one_hot_error(correct.into(),logits,sharpness)).sum::<f32>()/l
	}pub fn gelu_layer(&mut self,compact:bool,differentiate:bool){
		let layerlen=self.bufferlen;
		let alloccompact=if compact&!differentiate{
			self.bufferlen=0;
			false
		}else{compact};
		let (input,inputderivatives,output,outputderivatives,_,_,_,_)=self.allocate(alloccompact,differentiate,layerlen,0,0);
		if differentiate{rd_gelu_layer(input,inputderivatives,outputderivatives)}else if compact{gelu_layer(None,output)}else{gelu_layer(Some(input),output)}
	}pub fn gelu_multilayer_perceptron(&mut self,compact:bool,differentiate:bool,inputdimension:usize,intermediatedimension:usize,intermediatelayers:usize,outputdimension:usize){
		if intermediatelayers==0{return self.affine_layer(compact,differentiate,inputdimension,outputdimension)}
		if differentiate{
			self.affine_layer(compact,true,intermediatedimension,outputdimension);
			(0..intermediatelayers-1).for_each(|_|{
				self.gelu_layer(compact,true);
				self.affine_layer(compact,true,intermediatedimension,intermediatedimension);
			});
			self.gelu_layer(compact,true);
			self.affine_layer(compact,true,inputdimension,intermediatedimension);
		}else{
			self.affine_layer(compact,false,inputdimension,intermediatedimension);
			self.gelu_layer(compact,false);
			(0..intermediatelayers-1).for_each(|_|{
				self.affine_layer(compact,false,intermediatedimension,intermediatedimension);
				self.gelu_layer(compact,false);
			});
			self.affine_layer(compact,false,intermediatedimension,outputdimension);
		}
	}pub fn gpt<T:Copy+Into<usize>>(&mut self,compact:bool,contextdimension:usize,differentiate:bool,headcount:usize,headdimension:usize,priorlen:usize,transformerlayers:usize,tokens:&[T],uniquetokencount:usize){
		let embeddingarea=headcount*headdimension;
		if differentiate{
			self.affine_layer(compact,true,embeddingarea,uniquetokencount);
			self.normalization_layer(compact,true,embeddingarea);
			(0..transformerlayers).for_each(|_|self.transformer_layer(compact,contextdimension,true,headcount,headdimension));
			self.embed(compact,contextdimension,true,embeddingarea,priorlen,tokens,uniquetokencount);
		}else{
			self.embed(compact,contextdimension,false,embeddingarea,priorlen,tokens,uniquetokencount);
			(0..transformerlayers).for_each(|_|self.transformer_layer(compact,contextdimension,false,headcount,headdimension));
			self.normalization_layer(compact,false,embeddingarea);
			self.affine_layer(compact,false,embeddingarea,uniquetokencount);
		}
	}pub fn input(&mut self,compact:bool,differentiate:bool,x:&[f32]){
		let (_,_,output,_,_,_,_,_)=self.allocate(compact,differentiate,x.len(),0,0);
		if !differentiate{output.copy_from_slice(x)}
	}pub fn new()->Self{Self::default()}
	pub fn normalization_layer(&mut self,compact:bool,differentiate:bool,dimension:usize){
		const EPSILON:f32=1.0E-5;
		let (compactforward,len,li)=(compact&!differentiate,self.bufferlen,(dimension as f32).recip());
		let (alloccompact,arl,average_and_reciprocal_deviation)=(if compactforward{
			self.bufferlen=0;
			false
		}else{compact},len/dimension,|data:&[f32]|{
			let sum=data.iter().sum::<f32>();
			let average=li*sum;
			(average,(EPSILON+data.iter().map(|v|average-v).map(|v|v*v).sum::<f32>()*li).powf(-0.5))
		});
		let (input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,temp,_)=self.allocate(alloccompact,differentiate,len,dimension*2,if compactforward{0}else{arl*2});
		if compactforward{
			let (beta,gamma)=parameters.split_at(dimension);
			output.chunks_exact_mut(dimension).for_each(|data|{
				let (a,r)=average_and_reciprocal_deviation(data);
				beta.iter().zip(data).zip(gamma).for_each(|((b,v),g)|*v=(*v-a)*g*r+b);
			});
		}else if differentiate{
			let ((averages,reciprocaldeviations),(_,gamma),(betaderivatives,gammaderivatives))=(temp.split_at(arl),parameters.split_at(dimension),parametersderivatives.split_at_mut(dimension));
			averages.iter().zip(inputderivatives.chunks_exact_mut(dimension)).zip(outputderivatives.chunks_exact(dimension)).zip(reciprocaldeviations).zip(input.chunks_exact(dimension)).for_each(|((((a,dx),dy),r),x)|{
				let r2li=r*r*li;
				let ar2li=a*r2li;
				let dygysum=dy.iter().zip(gamma).map(|(dy,g)|dy*g).sum::<f32>();//TODO merge sum iterations
				let ar2lidygysum=ar2li*dygysum;
				let dygysumli=dygysum*li;
				let dygyxsumr2li=dy.iter().zip(gamma).zip(x).map(|((dy,g),x)|dy*g*x).sum::<f32>()*r2li;
				betaderivatives.iter_mut().zip(&mut *gammaderivatives).zip(dx.iter_mut()).zip(dy).zip(gamma).zip(x).for_each(|(((((db,dg),dx),dy),g),x)|{
					let xnma=x-a;
					*db+=dy;
					*dg+=dy*xnma*r;
					*dx=((ar2lidygysum-dygyxsumr2li)*xnma-dygysumli+dy*g)*r;
				});
			});
		}else{
			let ((averages,reciprocaldeviations),(beta,gamma))=(temp.split_at_mut(arl),parameters.split_at(dimension));
			averages.iter_mut().zip(reciprocaldeviations).zip(input.chunks_exact(dimension)).zip(output.chunks_exact_mut(dimension)).for_each(|(((average,reciprocaldeviation),x),y)|{
				let (a,r)=average_and_reciprocal_deviation(x);
				(*average,*reciprocaldeviation)=(a,r);
				beta.iter().zip(gamma).zip(x).zip(y).for_each(|(((b,g),x),y)|*y=(x-a)*g*r+b);
			});
		}
	}pub fn output(&self)->&[f32]{
		let (buffer,bufferlen,bufferoffset)=(&self.buffer,self.bufferlen,self.bufferoffset);
		&buffer[bufferoffset..bufferlen+bufferoffset]
	}pub fn projection_layer(&mut self,compact:bool,differentiate:bool,embeddingdimension:usize,intermediatedimension:usize){
		let (compactforward,iolen)=(compact&!differentiate,self.bufferlen);
		let intermediatelen=iolen/embeddingdimension*intermediatedimension;
		let (inputparameterslen,outputparameterslen)=((embeddingdimension+1)*intermediatedimension,(intermediatedimension+1)*embeddingdimension);
		let (input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)=self.allocate(compact,differentiate,iolen,inputparameterslen+outputparameterslen,if compactforward{1}else{2}*intermediatelen);
		let (inputparameters,outputparameters)=parameters.split_at(inputparameterslen);
		if differentiate{
			let ((inputparametersderivatives,outputparametersderivatives),(intermediatea,intermediateb),(intermediateaderivatives,intermediatebderivatives))=(parametersderivatives.split_at_mut(inputparameterslen),temp.split_at(intermediatelen),tempderivatives.split_at_mut(intermediatelen));
			inputderivatives.copy_from_slice(outputderivatives);
			intermediatebderivatives.fill(0.0);
			rd_trans_mat_mul(intermediatebderivatives,outputparametersderivatives,outputderivatives,ACC|BIAS,intermediateb,intermediatedimension,outputparameters);
			rd_gelu_layer(intermediatea,intermediateaderivatives,intermediatebderivatives);
			rd_trans_mat_mul(inputderivatives,inputparametersderivatives,intermediateaderivatives,BIAS,input,embeddingdimension,inputparameters);
		}else if compact{
			output.copy_from_slice(input);
			trans_mat_mul(BIAS,input,embeddingdimension,inputparameters,temp);
			gelu_layer(None,temp);
			trans_mat_mul(ACC|BIAS,temp,intermediatedimension,outputparameters,output);
		}else{
			let (intermediatea,intermediateb)=temp.split_at_mut(intermediatelen);
			output.copy_from_slice(input);
			trans_mat_mul(BIAS,input,embeddingdimension,inputparameters,intermediatea);
			gelu_layer(Some(intermediatea),intermediateb);
			trans_mat_mul(ACC|BIAS,intermediateb,intermediatedimension,outputparameters,output);
		}
	}pub fn reset(&mut self){(self.bufferlen,self.bufferoffset,self.parametersposition)=(0,0,0)}
	pub fn soft_choose(&mut self,lastn:usize,sharpness:f32)->usize{
		let (buffer,bufferlen,bufferoffset,seed)=(&self.buffer,self.bufferlen,self.bufferoffset,&mut self.seed);	
		soft_choose(&buffer[(bufferlen+bufferoffset).saturating_sub(lastn)..bufferlen+bufferoffset],seed,sharpness)
	}pub fn squared_error(&self,target:&[f32])->f32{squared_error(self.output(),target)}
	pub fn transformer_layer(&mut self,compact:bool,contextdimension:usize,differentiate:bool,headcount:usize,headdimension:usize){
		let embeddingarea=headcount*headdimension;
		let projectionarea=embeddingarea*4;
		if differentiate{
			self.projection_layer(compact,true,embeddingarea,projectionarea);
			self.normalization_layer(compact,true,embeddingarea);
			self.attention_layer(compact,contextdimension,true,headcount,headdimension);
			self.normalization_layer(compact,true,embeddingarea);
		}else{
			self.normalization_layer(compact,false,embeddingarea);
			self.attention_layer(compact,contextdimension,false,headcount,headdimension);
			self.normalization_layer(compact,false,embeddingarea);
			self.projection_layer(compact,false,embeddingarea,projectionarea);
		}
	}
}impl Default for AIBlock{
	fn default()->Self{Vec::new().into()}
}impl From<Vec<f32>>for AIBlock{
	fn from(v:Vec<f32>)->Self{Self::from_existing_data(Vec::new(),Vec::new(),0,0,0,v,Vec::new(),0,rseed())}
}impl GPT{
	fn from_existing_data(block:AIBlock,contextdimension:usize,headcount:usize,headdimension:usize,transformerlayers:usize,uniquetokencount:usize)->Self{
		Self{block,contextdimension,headcount,headdimension,transformerlayers,uniquetokencount}
	}pub fn choose_next<T:Copy+Into<usize>>(&mut self,sharpness:f32,tokens:&[T])->usize{//TODO bug
		let (block,contextdimension,headcount,headdimension,transformerlayers,uniquetokencount)=(&mut self.block,self.contextdimension,self.headcount,self.headdimension,self.transformerlayers,self.uniquetokencount);
		block.reset();
		block.gpt(true,contextdimension,false,headcount,headdimension,0,transformerlayers,tokens,uniquetokencount);
		block.soft_choose(uniquetokencount,sharpness)
	}pub fn evaluate<T:Copy+Into<usize>>(&mut self,examples:&[T])->f32{
		let (block,contextdimension,el,headcount,headdimension,transformerlayers,uniquetokencount)=(&mut self.block,self.contextdimension,examples.len()-1,self.headcount,self.headdimension,self.transformerlayers,self.uniquetokencount);
		block.reset();
		block.gpt(true,contextdimension,false,headcount,headdimension,0,transformerlayers,&examples[..el-1],uniquetokencount);
		block.entropic_one_hot_error(&examples[1..],uniquetokencount,1.0)
	}pub fn from_parameters(contextdimension:usize,headcount:usize,headdimension:usize,transformerlayers:usize,uniquetokencount:usize,parameters:Vec<f32>)->Self{Self::from_existing_data(parameters.into(),contextdimension,headcount,headdimension,transformerlayers,uniquetokencount)}
	pub fn generate_u8(&mut self,filldimension:&mut usize,sharpness:f32,tokens:&mut [u8]){//TODO be consistent about tokenization generics. get a better terminator system. also maybe this should have a sliding window or increase context dimension if filldimension gets too large. also compact is mainly to save memory on derivatives and there should be a non compact process option that reserves space so this isn't cubic
		let limit=tokens.len();
		while *filldimension<limit{
			let (past,future)=tokens.split_at_mut(*filldimension);
			*filldimension+=1;
			future[0]=self.choose_next(sharpness,past)as u8;
		}
	}pub fn new(contextdimension:usize,headcount:usize,headdimension:usize,transformerlayers:usize,uniquetokencount:usize)->Self{Self::from_parameters(contextdimension,headcount,headdimension,transformerlayers,uniquetokencount,position_encodings(headcount*headdimension).take(contextdimension*headcount*headdimension).collect())}
	pub fn resize_context(&mut self,contextdimension:usize){
		let (embeddingarea,parameters,priorcontextdimension)=(self.headcount*self.headdimension,&mut self.block.parameters,self.contextdimension);
		let (positionencodinglen,priorpositionencodinglen,priorparameterslen)=(contextdimension*embeddingarea,embeddingarea*priorcontextdimension,parameters.len());
		let parameterslen=priorparameterslen-priorpositionencodinglen+positionencodinglen;
		if contextdimension<priorcontextdimension{
			parameters.copy_within(priorpositionencodinglen..priorparameterslen,positionencodinglen);
			parameters.truncate(parameterslen);
		}else if contextdimension>priorcontextdimension{
			parameters.resize(parameterslen,0.0);
			parameters.copy_within(priorpositionencodinglen..priorparameterslen,positionencodinglen);
			parameters.iter_mut().zip(position_encodings(embeddingarea)).take(positionencodinglen).skip(priorpositionencodinglen).for_each(|(p,e)|*p=e);
		}self.contextdimension=contextdimension;
	}pub fn sample_string(&mut self)->String{
		let mut result=vec![32;self.contextdimension];
		self.generate_u8(&mut 0,1.0,&mut result);
		String::from_utf8_lossy(&result).to_string()
	}pub fn seed(&mut self)->&mut u128{&mut self.block.seed}
	pub fn train<T:Copy+Into<usize>>(&mut self,examples:&[T],persistence:f32,step:f32){
		let (block,contextdimension,el,headcount,headdimension,transformerlayers,uniquetokencount)=(&mut self.block,self.contextdimension,examples.len()-1,self.headcount,self.headdimension,self.transformerlayers,self.uniquetokencount);
		block.reset();
		block.gpt(false,contextdimension,false,headcount,headdimension,0,transformerlayers,&examples[..el-1],uniquetokencount);
		block.differentiate_entropic_one_hot_error(&examples[1..],-1.0,uniquetokencount,1.0);
		block.gpt(true,contextdimension,true,headcount,headdimension,0,transformerlayers,&examples[..el-1],uniquetokencount);
		block.adjust(persistence,step);
	}
}impl GeluMultilayerPerceptron{
	fn from_existing_data(block:AIBlock,inputdimension:usize,intermediatedimension:usize,intermediatelayers:usize,outputdimension:usize)->Self{
		Self{block,inputdimension,intermediatedimension,intermediatelayers,outputdimension}
	}pub fn from_parameters(inputdimension:usize,intermediatedimension:usize,intermediatelayers:usize,outputdimension:usize,parameters:Vec<f32>)->Self{Self::from_existing_data(parameters.into(),inputdimension,intermediatedimension,intermediatelayers,outputdimension)}
	pub fn new(inputdimension:usize,intermediatedimension:usize,intermediatelayers:usize,outputdimension:usize)->Self{Self::from_existing_data(AIBlock::new(),inputdimension,intermediatedimension,intermediatelayers,outputdimension)}
	pub fn process<'a>(&'a mut self,input:&[f32])->&'a [f32]{
		let (block,inputdimension,intermediatedimension,intermediatelayers,outputdimension)=(&mut self.block,self.inputdimension,self.intermediatedimension,self.intermediatelayers,self.outputdimension);
		block.reset();
		block.input(true,false,input);
		block.gelu_multilayer_perceptron(true,false,inputdimension,intermediatedimension,intermediatelayers,outputdimension);
		block.output()
	}pub fn train(&mut self,input:&[f32],iterations:usize,persistence:f32,step:f32,target:&[f32])->f32{
		let (block,inputdimension,intermediatedimension,intermediatelayers,il,outputdimension)=(&mut self.block,self.inputdimension,self.intermediatedimension,self.intermediatelayers,input.len(),self.outputdimension);
		block.reset();
		block.input(false,false,input);
		(0..iterations).for_each(|_|{
			block.gelu_multilayer_perceptron(false,false,inputdimension,intermediatedimension,intermediatelayers,outputdimension);
			block.differentiate_squared_error(-1.0,target);
			block.gelu_multilayer_perceptron(true,true,inputdimension,intermediatedimension,intermediatelayers,outputdimension);
			block.adjust(persistence,step);
		});
		block.gelu_multilayer_perceptron(false,false,inputdimension,intermediatedimension,intermediatelayers,outputdimension);
		block.squared_error(target)
	}
}impl SLT{
	fn from_existing_data(block:AIBlock,contextdimension:usize,embeddingdimension:usize,heads:usize,intermediatedimension:usize,outputdimension:usize)->Self{
		Self{block,contextdimension,embeddingdimension,heads,intermediatedimension,outputdimension}
	}pub fn from_parameters(contextdimension:usize,embeddingdimension:usize,heads:usize,intermediatedimension:usize,outputdimension:usize,parameters:Vec<f32>)->Self{Self::from_existing_data(parameters.into(),contextdimension,embeddingdimension,heads,intermediatedimension,outputdimension)}
	pub fn new(contextdimension:usize,embeddingdimension:usize,heads:usize,intermediatedimension:usize,outputdimension:usize)->Self{Self::from_parameters(contextdimension,embeddingdimension,heads,intermediatedimension,outputdimension,position_encodings(heads*intermediatedimension).take(contextdimension*heads*intermediatedimension).collect())}
	pub fn process<'a>(&'a mut self,input:&[u8])->&'a [f32]{
		let (block,contextdimension,embeddingdimension,heads,intermediatedimension,outputdimension)=(&mut self.block,self.contextdimension,self.embeddingdimension,self.heads,self.intermediatedimension,self.outputdimension);
		block.reset();
		block.embed(true,contextdimension,false,heads*intermediatedimension,0,input,256);
		block.transformer_layer(true,contextdimension,false,heads,intermediatedimension);
		block.normalization_layer(true,false,heads*intermediatedimension);
		block.affine_layer(true,false,heads*intermediatedimension,outputdimension);
		block.output()
	}pub fn train(&mut self,input:&[u8],iterations:usize,persistence:f32,step:f32,target:&[f32])->f32{
		let (block,contextdimension,embeddingdimension,heads,intermediatedimension,outputdimension)=(&mut self.block,self.contextdimension,self.embeddingdimension,self.heads,self.intermediatedimension,self.outputdimension);
		block.reset();
		(0..iterations).for_each(|_|{
			block.embed(false,contextdimension,false,heads*intermediatedimension,0,input,256);
			block.transformer_layer(false,contextdimension,false,heads,intermediatedimension);
			block.normalization_layer(false,false,heads*intermediatedimension);
			block.affine_layer(false,false,heads*intermediatedimension,outputdimension);
			block.differentiate_squared_error(-1.0,target);
			block.affine_layer(true,true,heads*intermediatedimension,outputdimension);
			block.normalization_layer(true,true,heads*intermediatedimension);
			block.transformer_layer(true,contextdimension,true,heads,intermediatedimension);
			block.embed(true,contextdimension,true,heads*intermediatedimension,0,input,256);
			block.adjust(persistence,step);
		});
		block.embed(false,contextdimension,false,heads*intermediatedimension,0,input,256);
		block.transformer_layer(false,contextdimension,false,heads,intermediatedimension);
		block.normalization_layer(false,false,heads*intermediatedimension);
		block.affine_layer(false,false,heads*intermediatedimension,outputdimension);
		block.squared_error(target)
	}
}impl Save for AIBlock{
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{Ok(Self::from_existing_data(Vec::read(reader)?,Vec::read(reader)?,usize::read(reader)?,usize::read(reader)?,usize::read(reader)?,Vec::read(reader)?,Vec::read(reader)?,usize::read(reader)?,u128::read(reader)?))}
	fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{
		self.buffer.write(writer)?;
		self.bufferderivatives.write(writer)?;
		self.bufferlen.write(writer)?;
		self.bufferoffset.write(writer)?;
		self.derivativesoffset.write(writer)?;
		self.parameters.write(writer)?;
		self.parametersderivatives.write(writer)?;
		self.parametersposition.write(writer)?;
		self.seed.write(writer)
	}
}impl Save for GPT{
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{Ok(Self::from_existing_data(AIBlock::read(reader)?,usize::read(reader)?,usize::read(reader)?,usize::read(reader)?,usize::read(reader)?,usize::read(reader)?))}
	fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{
		self.block.write(writer)?;
		self.contextdimension.write(writer)?;
		self.headcount.write(writer)?;
		self.headdimension.write(writer)?;
		self.transformerlayers.write(writer)?;
		self.uniquetokencount.write(writer)
	}
}pub const ACC:u8=1;
pub const BIAS:u8=2;
pub fn dot(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(a,b)|a*b).sum()}
pub fn entropic_one_hot_error(correct:usize,logits:&[f32],sharpness:f32)->f32{
	let lic=||logits.iter().copied();
	let m=if sharpness<0.0{lic().fold(f32::INFINITY,f32::min)}else{lic().fold(f32::NEG_INFINITY,f32::max)};
	let lnchances=||lic().map(|x|if m==x{0.0}else{(x-m)*sharpness});
	let lnsum=lnchances().map(f32::exp).sum::<f32>().ln();
	lnsum-lnchances().nth(correct).expect("correct bit is out of range")
}pub fn gelu(x:f32)->f32{(((x*x*x*0.044715+x)*0.79788456).tanh()+1.0)*x*0.5}//TODO magic numbers
pub fn gelu_layer(input:Option<&[f32]>,layer:&mut [f32]){
	if let Some(input)=input{input.iter().zip(layer).for_each(|(&x,y)|*y=gelu(x))}
	else{layer.iter_mut().for_each(|x|*x=gelu(*x))}
}pub fn position_encodings(embeddingdimension:usize)->impl Iterator<Item=f32>{
	let m=-2.0/(embeddingdimension as f32);
	(0..).map(move|x|{
		let y=x%embeddingdimension;
		let x=(x-y)as f32;
		let z=x*10000f32.powf((y/2)as f32*m);
		if y&1==0{z.sin()}else{z.cos()}
	})
}pub fn rd_dot(a:&[f32],b:&[f32],da:&mut [f32],db:&mut [f32],dc:f32){
	a.iter().zip(b).zip(da).zip(db).for_each(|(((a,b),da),db)|{
		*da+=b*dc;
		*db+=a*dc;
	});
}pub fn rd_gelu(x:f32)->f32{
	let x2=x*x;
	let t=(x2*x*0.044715+x)*0.79788456;//TODO magic numbers
	let b=t.cosh().recip();
	((x2*0.134145+1.0)*b*b*x*0.79788456+t.tanh()+1.0)*0.5
}pub fn rd_gelu_layer(input:&[f32],inputderivatives:&mut [f32],layerderivatives:&[f32]){input.iter().zip(inputderivatives).zip(layerderivatives).for_each(|((&x,dx),dy)|*dx=dy*rd_gelu(x))}
pub fn rd_entropic_one_hot_error(correct:usize,errorderivative:f32,logits:&[f32],logitsderivatives:&mut [f32],sharpness:f32){//TODO since we probably don't want to make logitsderivatives acc, performance could be improved without too much effort byt using logitsderivatives as scratch space rather than recomputing chances
	let lic=||logits.iter().copied();
	let m=if sharpness<0.0{lic().fold(f32::INFINITY,f32::min)}else{lic().fold(f32::NEG_INFINITY,f32::max)};
	let chances=||lic().map(|x|if m==x{1.0}else{((x-m)*sharpness).exp()});
	let r=chances().sum::<f32>().recip()*errorderivative*sharpness;
	chances().zip(&mut *logitsderivatives).for_each(|(chance,dl)|*dl=chance*r);
	logitsderivatives[correct]-=errorderivative*sharpness;
}pub fn rd_squared_error(errorderivative:f32,logits:&[f32],logitsderivatives:&mut [f32],target:&[f32]){logits.iter().zip(logitsderivatives).zip(target).for_each(|((l,d),t)|*d=(l-t)*errorderivative*2.0)}
pub fn rd_trans_mat_mul(dx:&mut [f32],dy:&mut [f32],dz:&[f32],flags:u8,x:&[f32],xdimension:usize,y:&[f32]){
	let bias=BIAS&flags!=0;
	let ydimension=bias as usize+xdimension;
	let zdimension=y.len()/ydimension;
	dx.chunks_exact_mut(xdimension).zip(x.chunks_exact(xdimension)).zip(dz.chunks_exact(zdimension)).for_each(|((dx,x),dz)|dy.chunks_exact_mut(ydimension).zip(dz).zip(y.chunks_exact(ydimension)).for_each(|((dy,&dz),y)|{
		if bias{dy[0]+=dz}
		rd_dot(x,&y[bias as usize..],dx,&mut dy[bias as usize..],dz);
	}));
}pub fn rfloat(seed:&mut u128)->f32{
	update_seed(seed);
	*seed as i32 as f32/0x7FFFFFFF as f32
}pub fn rsize(bound:usize,seed:&mut u128)->usize{
	let mask=bound.next_power_of_two()-1;
	loop{
		update_seed(seed);
		let result=(*seed as usize)&mask;
		if bound>result{break result}
	}
}pub fn rseed()->u128{SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()}
pub fn soft_choose(logits:&[f32],seed:&mut u128,sharpness:f32)->usize{
	let lic=||logits.iter().copied();
	let m=if sharpness<0.0{lic().fold(f32::INFINITY,f32::min)}else{lic().fold(f32::NEG_INFINITY,f32::max)};
	let chances=||lic().map(|x|if m==x{1.0}else{((x-m)*sharpness).exp()});
	let mut choice=chances().sum::<f32>()*rfloat(seed);
	for (n,chance)in chances().enumerate(){
		choice-=chance;
		if choice<0.0{return n}
	}0
}pub fn squared_error(logits:&[f32],target:&[f32])->f32{logits.iter().zip(target).map(|(l,t)|l-t).map(|x|x*x).sum()}
pub fn trans_mat_mul(flags:u8,x:&[f32],xdimension:usize,y:&[f32],z:&mut [f32]){
	let (acc,bias)=(ACC&flags!=0,BIAS&flags!=0);
	let ydimension=bias as usize+xdimension;
	let zdimension=y.len()/ydimension;
	x.chunks_exact(xdimension).zip(z.chunks_exact_mut(zdimension)).for_each(|(x,z)|y.chunks_exact(ydimension).zip(z).for_each(|(y,z)|*z=dot(&y[bias as usize..],x)+if acc{*z}else{0.0}+if bias{y[0]}else{0.0}));
}pub fn update_seed(seed:&mut u128){
	*seed^=*seed<<15;
	*seed^=*seed>>4;
	*seed^=*seed<<21;
}
