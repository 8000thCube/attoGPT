use crate::save::Save;//TODO improve documentation
use std::{
	io::{Error as IOError,Read as IORead,Write as IOWrite},
	iter::once,
	time::{SystemTime,UNIX_EPOCH}
};
#[derive(Clone,Debug,Default)]
/// A struct on which to invoke layer oriented neural network methods
pub struct NeuralNetwork{pub buffer:Vec<f32>,pub parameters:Vec<f32>}
#[derive(Clone,Debug)]
/// Multilayer perceptron with gelu activation function on inner layers
pub struct GeluMultilayerPerceptron{network:NeuralNetwork,pub bias:bool,pub inputdimension:usize,pub intermediatedimension:usize,pub intermediatelayercount:usize,pub outputdimension:usize}
#[derive(Clone,Debug)]
/// Transformer with 1 layer to test the attention function
pub struct SLT{network:NeuralNetwork,pub contextdimension:usize,pub heads:usize,pub intermediatedimension:usize,pub outputdimension:usize,uniquetokencount:usize}
#[derive(Clone,Debug)]
/// GPT style transformer
pub struct GPT{network:NeuralNetwork,pub bias:bool,pub blockcount:usize,pub headcount:usize,pub headdimension:usize,pub projectiondimension:usize,pub uniquetokencount:usize}
impl GPT{
	pub fn evaluate<C:Clone+ExactSizeIterator<Item=usize>>(&mut self,sharpness:f32,tokens:C)->f32{
		let (bias,blockcount,headcount,headdimension,network,projectiondimension,uniquetokencount)=(self.bias,self.blockcount,self.headcount,self.headdimension,&mut self.network,self.projectiondimension,self.uniquetokencount);
		let (embeddingarea,mut compactionheight,mut offset,mut parametersposition,tl,tokens)=(headcount*headdimension,0,0,0,tokens.len(),||tokens.clone());
		if tl==0{return f32::NAN}
		let contextdimension=tl-1;
		let mut len=contextdimension*embeddingarea;
		network.force_state(len);
		network.state_mut(len,offset).iter_mut().zip(position_encodings(embeddingarea)).for_each(|(x,p)|*x=p);
		network.gpt(bias,blockcount,&mut compactionheight,contextdimension,None,None,headcount,headdimension,projectiondimension,(0..contextdimension).map(|position|((0..position+1),position)),&mut len,&mut offset,&mut parametersposition,tokens().take(contextdimension),uniquetokencount);
		network.state(len,offset).chunks_exact(uniquetokencount).zip(tokens().skip(1)).map(|(logits,token)|entropic_one_hot_error(token,logits,sharpness)).sum::<f32>()/contextdimension as f32
	}pub fn from_parameters(bias:bool,blockcount:usize,headcount:usize,headdimension:usize,parameters:Vec<f32>,projectiondimension:usize,uniquetokencount:usize)->Self{
		Self{bias,blockcount,headcount,headdimension,network:NeuralNetwork::from_parameters(parameters),projectiondimension,uniquetokencount}
	}pub fn infer<'a,E:ExactSizeIterator<Item=usize>>(&'a mut self,emptycontextdimension:usize,sharpness:f32,tokens:E)->impl 'a+Iterator<Item=usize>{
		let (bias,blockcount,headcount,headdimension,network,projectiondimension,uniquetokencount)=(self.bias,self.blockcount,self.headcount,self.headdimension,&mut self.network,self.projectiondimension,self.uniquetokencount);
		let (embeddingarea,filledcontextdimension,mut compactionheight,mut offset,mut parametersposition,mut seed)=(headcount*headdimension,tokens.len(),0,0,0,rseed());
		let (contextdimension,mut len,mut positionencodings)=(emptycontextdimension+filledcontextdimension,embeddingarea,position_encodings(embeddingarea));
		network.force_state(len);
		(0..filledcontextdimension).map(|position|((0..position+1),position)).zip(tokens).for_each(|(maskitervalue,token)|{
			(compactionheight,len,offset,parametersposition)=(0,embeddingarea,0,0);
			network.state_mut(len,offset).iter_mut().for_each(|x|*x=positionencodings.next().unwrap());
			network.gpt(bias,blockcount,&mut compactionheight,contextdimension,None,None,headcount,headdimension,projectiondimension,once(maskitervalue),&mut len,&mut offset,&mut parametersposition,once(token),uniquetokencount);
		});
		let mut token=soft_choose(network.state_mut(len,offset),&mut seed,sharpness);
		(filledcontextdimension..contextdimension).map(|position|((0..position+1),position)).chain((0..contextdimension).cycle().map(move|position|((0..contextdimension),position))).map(move|maskitervalue|{
			let (mut compactionheight,mut len,mut offset,mut parametersposition,t)=(0,embeddingarea,0,0,token);
			network.state_mut(len,offset).iter_mut().for_each(|x|*x=positionencodings.next().unwrap());
			network.gpt(bias,blockcount,&mut compactionheight,contextdimension,None,None,headcount,headdimension,projectiondimension,once(maskitervalue),&mut len,&mut offset,&mut parametersposition,once(t),uniquetokencount);
			token=soft_choose(network.state_mut(len,offset),&mut seed,sharpness);
			t
		})
	}pub fn new(bias:bool,blockcount:usize,headcount:usize,headdimension:usize,projectiondimension:usize,uniquetokencount:usize)->Self{Self::from_parameters(bias,blockcount,headcount,headdimension,Vec::new(),projectiondimension,uniquetokencount)}
	pub fn sample_string(&mut self,contextdimension:usize,len:usize,sharpness:f32)->String{String::from_utf8_lossy(&self.infer(contextdimension.saturating_sub(1),sharpness,once('\n' as usize)).map(|x|x as u8).take(len).collect::<Vec<u8>>()).to_string()}
	pub fn train<C:Clone+Iterator<Item=usize>,F:FnMut(&mut NeuralNetwork,&mut NeuralNetwork)>(&mut self,batchdimension:usize,exampledimension:usize,networkderivatives:Option<&mut NeuralNetwork>,mut optimizer_step:F,sharpness:f32,mut sequence:C){// f(network, networkderivatives), example dimension should be one more than context dimension
		let (bias,blockcount,headcount,headdimension,network,projectiondimension,uniquetokencount)=(self.bias,self.blockcount,self.headcount,self.headdimension,&mut self.network,self.projectiondimension,self.uniquetokencount);
		assert!(batchdimension!=0);
		assert!(exampledimension>1);
		let (batcharea,contextdimension,embeddingarea,mut defaultnetworkderivatives)=(batchdimension*exampledimension,exampledimension-1,headcount*headdimension,NeuralNetwork::default());
		let (mut len,mut offset,mut parametersposition,networkderivatives,itlen)=(batchdimension*contextdimension*embeddingarea,0,0,networkderivatives.unwrap_or(&mut defaultnetworkderivatives),batcharea-batchdimension);
		network.force_state(len);
		network.state_mut(len,0).iter_mut().zip(position_encodings(embeddingarea)).for_each(|(x,p)|*x=p);
		networkderivatives.force_state(batchdimension*contextdimension*uniquetokencount);
		while sequence.clone().next().is_some(){
			let (batch_examples_skipping_nths_flattened,mut compactionheight,mut derivativesoffset)=(|n:usize|{
				let mut enumeratedtokens=sequence.clone().enumerate().take(batcharea);      
				(0..itlen).map(move|_|{
					let (k,token)=enumeratedtokens.next().unwrap();
					(k%exampledimension==n).then(||{
						let (_,token)=enumeratedtokens.next().unwrap();
						token
					}).unwrap_or(token)
				})
			},usize::MAX,0);
			let (input,target)=(||batch_examples_skipping_nths_flattened(contextdimension),||batch_examples_skipping_nths_flattened(0));
			network.gpt(bias,blockcount,&mut compactionheight,contextdimension,None,None,headcount,headdimension,projectiondimension,(0..contextdimension).map(|position|((0..position+1),position)),&mut len,&mut offset,&mut parametersposition,input(),uniquetokencount);
			compactionheight=0;
			let (output,outputderivatives)=(network.state(len,offset),networkderivatives.state_mut(len,derivativesoffset));
			output.chunks_exact(uniquetokencount).zip(outputderivatives.chunks_exact_mut(uniquetokencount)).zip(target()).for_each(|((logits,logitsderivatives),token)|rd_entropic_one_hot_error(token,-1.0,logits,logitsderivatives,sharpness));
			network.gpt(bias,blockcount,&mut compactionheight,contextdimension,Some(networkderivatives),Some(&mut derivativesoffset),headcount,headdimension,projectiondimension,(0..contextdimension).map(|position|((0..position+1),position)),&mut len,&mut offset,&mut parametersposition,input(),uniquetokencount);
			optimizer_step(network,networkderivatives);
			sequence.nth(batcharea-1);
		}
	}
}impl GeluMultilayerPerceptron{
	pub fn from_parameters(bias:bool,inputdimension:usize,intermediatedimension:usize,intermediatelayercount:usize,outputdimension:usize,parameters:Vec<f32>)->Self{
		Self{bias,inputdimension,intermediatedimension,intermediatelayercount,network:NeuralNetwork::from_parameters(parameters),outputdimension}
	}pub fn new(bias:bool,inputdimension:usize,intermediatedimension:usize,intermediatelayercount:usize,outputdimension:usize)->Self{Self::from_parameters(bias,inputdimension,intermediatedimension,intermediatelayercount,outputdimension,Vec::new())}
	pub fn process<'a>(&'a mut self,input:&[f32])->&'a [f32]{
		let (bias,inputdimension,intermediatedimension,intermediatelayercount,network,outputdimension)=(self.bias,self.inputdimension,self.intermediatedimension,self.intermediatelayercount,&mut self.network,self.outputdimension);
		let (mut len,mut offset)=(input.len(),0);
		network.force_state(len);
		network.state_mut(len,offset).copy_from_slice(input);
		network.gelu_multilayer_perceptron(bias,0,None,None,inputdimension,intermediatedimension,intermediatelayercount,&mut len,&mut offset,outputdimension,&mut 0);
		network.state(len,offset)
	}pub fn train(&mut self,derivatives:Option<&mut NeuralNetwork>,input:&[f32],iterations:usize,momentum:f32,rate:f32,target:&[f32]){
		let (bias,inputdimension,intermediatedimension,intermediatelayercount,network,outputdimension)=(self.bias,self.inputdimension,self.intermediatedimension,self.intermediatelayercount,&mut self.network,self.outputdimension);
		let (mut defaultderivatives,mut derivativesoffset,mut len,mut offset,mut parametersposition)=(NeuralNetwork::default(),0,input.len(),0,0);
		let derivatives=derivatives.unwrap_or(&mut defaultderivatives);
		derivatives.force_state(len/inputdimension*outputdimension);
		network.force_state(len);
		network.state_mut(len,offset).copy_from_slice(input);
		(0..iterations).for_each(|_|{
			network.gelu_multilayer_perceptron(bias,usize::MAX,None,None,inputdimension,intermediatedimension,intermediatelayercount,&mut len,&mut offset,outputdimension,&mut parametersposition);
			let (output,outputderivatives)=(network.state(len,offset),derivatives.state_mut(len,derivativesoffset));
			rd_squared_error(-1.0,output,outputderivatives,target);
			network.gelu_multilayer_perceptron(bias,0,Some(derivatives),Some(&mut derivativesoffset),inputdimension,intermediatedimension,intermediatelayercount,&mut len,&mut offset,outputdimension,&mut parametersposition);
			network.msgd(derivatives,momentum,rate);
		});
	}
}impl SLT{
	pub fn from_parameters(contextdimension:usize,heads:usize,intermediatedimension:usize,outputdimension:usize,parameters:Vec<f32>,uniquetokencount:usize)->Self{
		Self{contextdimension,heads,intermediatedimension,network:NeuralNetwork::from_parameters(parameters),outputdimension,uniquetokencount}
	}pub fn new(contextdimension:usize,heads:usize,intermediatedimension:usize,outputdimension:usize,uniquetokencount:usize)->Self{Self::from_parameters(contextdimension,heads,intermediatedimension,outputdimension,Vec::new(),uniquetokencount)}
	pub fn process<'a,E:ExactSizeIterator<Item=usize>>(&'a mut self,tokens:E)->&'a [f32]{
		let (contextdimension,heads,intermediatedimension,network,outputdimension,uniquetokencount)=(self.contextdimension,self.heads,self.intermediatedimension,&mut self.network,self.outputdimension,self.uniquetokencount);
		let (embeddingarea,filledcontextdimension,mut compactionheight,mut offset,mut parametersposition)=(heads*intermediatedimension,tokens.len(),0,0,0);
		let mut len=embeddingarea*filledcontextdimension;
		network.force_state(len);
		network.state_mut(len,offset).iter_mut().zip(position_encodings(embeddingarea)).for_each(|(x,p)|*x=p);
		network.embed(compactionheight,None,None,embeddingarea,len,&mut offset,&mut parametersposition,tokens,uniquetokencount);
		network.transformer_block(true,&mut compactionheight,contextdimension,None,None,heads,intermediatedimension,embeddingarea*4,(0..filledcontextdimension).map(|position|((0..position+1),position)),len,&mut offset,&mut parametersposition);
		network.normalization_layer(compactionheight,None,None,embeddingarea,len,&mut offset,&mut parametersposition);
		network.affine_layer(true,compactionheight,None,None,embeddingarea,&mut len,&mut offset,outputdimension,&mut parametersposition);
		network.state(len,offset)
	}pub fn train<C:Clone+ExactSizeIterator<Item=usize>>(&mut self,derivatives:Option<&mut NeuralNetwork>,input:C,iterations:usize,momentum:f32,rate:f32,target:&[f32]){
		let (contextdimension,heads,intermediatedimension,network,outputdimension,uniquetokencount)=(self.contextdimension,self.heads,self.intermediatedimension,&mut self.network,self.outputdimension,self.uniquetokencount);
		let (embeddingarea,input,mut compactionheight,mut defaultderivatives,mut derivativesoffset,mut offset,mut parametersposition)=(heads*intermediatedimension,||input.clone(),0,NeuralNetwork::default(),0,0,0);
		let (derivatives,mut len)=(derivatives.unwrap_or(&mut defaultderivatives),embeddingarea*input().len());
		derivatives.force_state(len/embeddingarea*outputdimension);
		network.force_state(len);
		network.state_mut(len,offset).iter_mut().zip(position_encodings(embeddingarea).take(contextdimension*embeddingarea).cycle()).for_each(|(x,p)|*x=p);
		(0..iterations).for_each(|_|{
			compactionheight=usize::MAX;
			network.embed(compactionheight,None,None,embeddingarea,len,&mut offset,&mut parametersposition,input(),uniquetokencount);
			network.transformer_block(true,&mut compactionheight,contextdimension,None,None,heads,intermediatedimension,embeddingarea*4,(0..contextdimension).map(|position|((0..position+1),position)),len,&mut offset,&mut parametersposition);
			network.normalization_layer(compactionheight,None,None,embeddingarea,len,&mut offset,&mut parametersposition);
			network.affine_layer(true,compactionheight,None,None,embeddingarea,&mut len,&mut offset,outputdimension,&mut parametersposition);
			compactionheight=0;
			let (output,outputderivatives)=(network.state(len,offset),derivatives.state_mut(len,derivativesoffset));
			rd_squared_error(-1.0,output,outputderivatives,target);
			network.affine_layer(true,compactionheight,Some(derivatives),Some(&mut derivativesoffset),embeddingarea,&mut len,&mut offset,outputdimension,&mut parametersposition);
			network.normalization_layer(compactionheight,Some(derivatives),Some(&mut derivativesoffset),embeddingarea,len,&mut offset,&mut parametersposition);
			network.transformer_block(true,&mut compactionheight,contextdimension,Some(derivatives),Some(&mut derivativesoffset),heads,intermediatedimension,embeddingarea*4,(0..contextdimension).map(|position|((0..position+1),position)),len,&mut offset,&mut parametersposition);
			network.embed(compactionheight,Some(derivatives),Some(&mut derivativesoffset),embeddingarea,len,&mut offset,&mut parametersposition,input(),uniquetokencount);
			network.msgd(derivatives,momentum,rate);
		});
	}
}impl Save for GPT{
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{Ok(Self::from_parameters(bool::read(reader)?,usize::read(reader)?,usize::read(reader)?,usize::read(reader)?,Vec::read(reader)?,usize::read(reader)?,usize::read(reader)?))}
	fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{
		self.bias.write(writer)?;
		self.blockcount.write(writer)?;
		self.headcount.write(writer)?;
		self.headdimension.write(writer)?;
		self.network.parameters.write(writer)?;
		self.projectiondimension.write(writer)?;
		self.uniquetokencount.write(writer)
	}
}/// Macro for concisely splitting a slice into multiple contiguous subslices of specific lengths
macro_rules! multi_split{
	($slice:expr,$splitfn:ident,$($lengths:expr),*)=>({
		let mut slice=$slice;
		let (result,_)=(($({
			let (a,b)=slice.$splitfn($lengths);
			slice=b;
			a
		}),*),slice);
		result
	})
}impl NeuralNetwork{
	/// Adaptive gradient optimization with weight decay
	pub fn adamw(&mut self,b1:f32,b2:f32,derivatives:&mut Self,rate:f32,step:usize,weightdecay:f32){
		const EPSILON:f32=1.0E-8;
		let step=step.clamp(1,i32::MAX as usize)as i32;
		let (b1hr,b2h,derivatives,parameters,weightpersistence)=((1.0-b1.powi(step)).recip()*rate,(1.0-b2.powi(step)).recip(),&mut derivatives.parameters,&mut self.parameters,1.0-rate*weightdecay);
		let pl=parameters.len();
		if derivatives.len()<pl*3{derivatives.resize(pl*3,0.0)}
		let (derivatives,mv)=derivatives.split_at_mut(pl);
		let (m,v)=mv.split_at_mut(pl);
		derivatives.iter_mut().zip(m).zip(v).zip(parameters).for_each(|(((d,m),v),x)|{
			(*m,*v)=((1.0-b1)**d+b1**m,(1.0-b2)**d**d+b2**v);
			(*d,*x)=(0.0,((b2h**v).sqrt()+EPSILON).recip()*b1hr**m+weightpersistence**x);//TODO this method of avoiding division by 0 is questionable. Similarly for the layer normalization. An infinite value would propogate and destroy progress, but I'm not sure a factor of 100000000 is very good either. Pytorch's version of the function, whose documentation I used for reference, does something similar, but perhaps some alternative such as clamping to a range, or mapping infinity to 0, should be tested.
		});
	}/// Applies a matrix transformation with optional bias, or differentiates it if derivatives and derivativesoffset are given. Buffer indices less than the minimum of compactionheight and offset will not be affected.
	pub fn affine_layer(&mut self,bias:bool,compactionheight:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,inputdimension:usize,len:&mut usize,offset:&mut usize,outputdimension:usize,parametersposition:&mut usize){
		let (flags,parameterslen)=(BIAS*bias as u8,(bias as usize+inputdimension)*outputdimension);
		*len=if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			let inputlen=*len/outputdimension*inputdimension;
			let (input,inputderivatives,_,outputderivatives,parameters,parametersderivatives,_,_)=self.allocate_layer_derivatives(compactionheight,derivatives,derivativesoffset,inputlen,offset,*len,parameterslen,parametersposition,0);
			inputderivatives.fill(0.0);
			rd_trans_mat_mul(inputderivatives,parametersderivatives,outputderivatives,flags,input,inputdimension,parameters);
			inputlen
		}else{
			let outputlen=*len/inputdimension*outputdimension;
			let (input,output,parameters,_)=self.allocate_layer(compactionheight,*len,offset,outputlen,parameterslen,parametersposition,0);
			trans_mat_mul(flags,input,inputdimension,parameters,output);
			outputlen
		};
	}/// Allocates or retrieves an input layer, an output layer, parameters, and temp. The layers are not necessarily zeroed; the may contain data that was previously in an overlapping range of the internal buffer. Buffer indices less than the minimum of compactionheight and offset will not be part of the returned slices. [input, temp, output] will be contiguous if *offset<compactionheight. To avoid compaction and just stack layers on top of each other like this, compactionheight can be set to usize::MAX. The offset before the method is called is the offset of the input layer. Afterwards, it is the offset of the output layer. Return order (input,output,parameters,temp)
	pub fn allocate_layer<'a>(&'a mut self,compactionheight:usize,inputlen:usize,offset:&mut usize,outputlen:usize,parameterslen:usize,parametersposition:&mut usize,templen:usize)->(&'a mut [f32],&'a mut [f32],&'a mut [f32],&'a mut [f32]){
		let (buffer,parameters)=(&mut self.buffer,&mut self.parameters);																															//grab relevant instance variables
		let (inputstart,parametersstart)=(*offset,*parametersposition);
		*parametersposition+=parameterslen;
		let (alloclen,compactstart,inputstop,parametersstop)=(outputlen+templen,inputstart.checked_sub(compactionheight).map(|x|x as isize).unwrap_or(-1),inputlen+inputstart,*parametersposition);	//find out how much length must be allocated as opposed to retrieve, and find the offset between the start of the compact region and the start of the input layer. Letting it be negative if the input layer is before the compact region conveniently skips the compact branches, and since its exact value is not needed in this case, just making it -1 is fine and the checked_sub conveniently allows compactionheight==usize::MAX for no compaction. Also find out where the input layer stops and where parameters stop.
		if parameters.len()<parametersstop{
			let mut seed=rseed();
			parameters.resize_with(parametersstop,||rfloat(&mut seed));
		}let parameters=&mut parameters[parametersstart..parametersstop];
		if alloclen as isize<=compactstart{																																							//check if the to be allocated length fits between the start of the compact region and the start of the input layer.
			*offset=compactionheight;
			let (output,temp,_,input)=multi_split!(&mut buffer[compactionheight..],split_at_mut,outputlen,templen,compactstart as usize-alloclen,inputlen);
			(input,output,parameters,temp)
		}else if compactstart>=outputlen as isize{																																					//if output and temp don't both fit, try just one of them.
			*offset=compactionheight;
			let tempstop=inputstop+templen;
			if buffer.len()<tempstop{buffer.resize(tempstop,0.0)}																																	//temp is allocated after input, so it might not fit in the buffer yet. Ensure the buffer's len so it does.
			let (output,_,input,temp)=multi_split!(&mut buffer[compactionheight..],split_at_mut,outputlen,compactstart as usize-outputlen,inputlen,templen);
			(input,output,parameters,temp)
		}else if compactstart>=templen as isize{
			*offset=inputstop;
			let outputstop=inputstop+outputlen;
			if buffer.len()<outputstop{buffer.resize(outputstop,0.0)}																																//output is allocated after input, so it might not fit in the buffer yet. Ensure the buffer's len so it does.
			let (temp,_,input,output)=multi_split!(&mut buffer[compactionheight..],split_at_mut,templen,compactstart as usize-templen,inputlen,outputlen);
			(input,output,parameters,temp)
		}else{																																														//neither fit. allocate uncompactly
			*offset=inputstop+templen;
			let outputstop=*offset+outputlen;
			if buffer.len()<outputstop{buffer.resize(outputstop,0.0)}																																//output is allocated after input, so it might not fit in the buffer yet. Ensure the buffer's len so it does. Temp is also, but it is before output, so it is not individually a concern.
			let (input,temp,output)=multi_split!(&mut buffer[inputstart..],split_at_mut,inputlen,templen,outputlen);
			(input,output,parameters,temp)
		}
	}/// Retrieves an input layer, an output layer, parameters, and temp, and allocates derivatives for the layers. Assumes compactionheight during forward pass was >= the final offset. Internally uses allocate_layer for the derivatives, so the caveats for that function apply. The compaction height in this function applies to the derivatives, since the layers have already been allocated. Return order (input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)
	pub fn allocate_layer_derivatives<'a,'b>(&'a mut self,compactionheight:usize,derivatives:&'b mut Self,derivativesoffset:&mut usize,inputlen:usize,offset:&mut usize,outputlen:usize,parameterslen:usize,parametersposition:&mut usize,templen:usize)->(&'a [f32],&'b mut [f32],&'a [f32],&'b mut [f32],&'a [f32],&'b mut [f32],&'a [f32],&'b mut [f32]){
		let (buffer,parameters)=(&self.buffer,&self.parameters);
		*offset-=inputlen+templen;
		let parametersstop=*parametersposition;
		*parametersposition-=parameterslen;
		let parametersstart=*parametersposition;
		let ((input,temp,output),(outputderivatives,inputderivatives,parametersderivatives,tempderivatives),parameters)=(multi_split!(&buffer[*offset..],split_at,inputlen,templen,outputlen),derivatives.allocate_layer(compactionheight,outputlen,derivativesoffset,inputlen,parameterslen,&mut parametersstart.clone(),templen),&parameters[parametersstart..parametersstop]);	//io for derivatives is swapped since this is reverse differentiation
		(input,inputderivatives,output,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)
	}/// a t t e n d //maskiter gives the positions and positions to which they attend. Caches keys and values so as long as context dimension doesn't change the same neural network operations could be run with different positions in the mask iter to attend to a sequence incrementally
	pub fn attention_layer<C:Clone+ExactSizeIterator<Item=(D,usize)>,D:Clone+ExactSizeIterator<Item=usize>>(&mut self,compactionheight:&mut usize,contextdimension:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,headcount:usize,headdimension:usize,len:usize,maskiter:C,offset:&mut usize,parametersposition:&mut usize){
		let (embeddingarea,mask_iter,newdimension,sharpness)=(headcount*headdimension,||maskiter.clone(),maskiter.len(),(headdimension as f32).recip().sqrt());
		let (attentionvolume,contextvolume,head_range,kv_range,kqvparametercount,newvolume)=(headcount*mask_iter().map(|(attendto,_)|attendto.len()).sum::<usize>(),contextdimension*embeddingarea,|head:usize,position:usize|{
			let start=embeddingarea*position+head*headdimension;
			let stop=headdimension+start;
			move||start..stop
		},|positionnumber:usize|{
			let start=embeddingarea*positionnumber;
			let stop=embeddingarea+start;
			move||start..stop
		},embeddingarea*embeddingarea,embeddingarea*newdimension);
		let (batchdimension,oparametercount)=(len/newvolume,embeddingarea+kqvparametercount);
		let (attentionlen,contextlen,parameterslen)=(attentionvolume*batchdimension,batchdimension*contextvolume,kqvparametercount*3+oparametercount);
		let (cachedtemplen,uncachedtemplen)=(contextlen*2,attentionlen+len*2);				//cache keys and values
		let templen=cachedtemplen+uncachedtemplen;
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			*compactionheight=(*compactionheight).max(*derivativesoffset+cachedtemplen+len);//move compactionheight to the beginning of uncached temp, which is the end of cached temp
			let (input,inputderivatives,_,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)=self.allocate_layer_derivatives(*compactionheight,derivatives,derivativesoffset,len,offset,len,parameterslen,parametersposition,templen);
			inputderivatives.copy_from_slice(outputderivatives);
			let ((keyparameters,outputparameters,queryparameters,valueparameters),(keyparametersderivatives,outputparametersderivatives,queryparametersderivatives,valueparametersderivatives),(keys,values,attention,queries,summaries),(keysderivatives,valuesderivatives,attentionderivatives,queriesderivatives,summariesderivatives))=(multi_split!(parameters,split_at,kqvparametercount,oparametercount,kqvparametercount,kqvparametercount),multi_split!(parametersderivatives,split_at_mut,kqvparametercount,oparametercount,kqvparametercount,kqvparametercount),multi_split!(temp,split_at,contextlen,contextlen,attentionlen,len,len),multi_split!(tempderivatives,split_at_mut,contextlen,contextlen,attentionlen,len,len));
			queriesderivatives.fill(0.0);
			summariesderivatives.fill(0.0);
			rd_trans_mat_mul(summariesderivatives,outputparametersderivatives,outputderivatives,ACC|BIAS,summaries,embeddingarea,outputparameters);
			attention.chunks_exact(attentionvolume).zip(attentionderivatives.chunks_exact_mut(attentionvolume)).zip(input.chunks_exact(newvolume)).zip(inputderivatives.chunks_exact_mut(newvolume)).zip(keys.chunks_exact(contextvolume)).zip(keysderivatives.chunks_exact_mut(contextvolume)).zip(queries.chunks_exact(newvolume)).zip(queriesderivatives.chunks_exact_mut(newvolume)).zip(summariesderivatives.chunks_exact(newvolume)).zip(values.chunks_exact(contextvolume)).zip(valuesderivatives.chunks_exact_mut(contextvolume)).for_each(|((((((((((attention,attentionderivatives),input),inputderivatives),keys),keysderivatives),queries),queriesderivatives),summariesderivatives),values),valuesderivatives)|{
				mask_iter().map(|(_,position)|kv_range(position)).for_each(|range|{
					(keysderivatives[range()].fill(0.0),valuesderivatives[range()].fill(0.0));
				});
				let mut attentionstart=0;
				mask_iter().enumerate().for_each(|(currentpositionindex,(attendto,_))|{
					let (attend_to_ranges,attentiondimension)=(|head:usize|attendto.clone().map(move|attendtoposition|head_range(head,attendtoposition)),attendto.len());
					let attentionarea=attentiondimension*headcount;
					let attentionstop=attentionarea+attentionstart;
					let (attention,attentionderivatives)=(&attention[attentionstart..attentionstop],&mut attentionderivatives[attentionstart..attentionstop]);
					attention.chunks_exact(attentiondimension).enumerate().zip(attentionderivatives.chunks_exact_mut(attentiondimension)).for_each(|((head,attention),attentionderivatives)|{
						let current_head_position_range=head_range(head,currentpositionindex);
						let (query,queryderivatives,summaryvaluederivatives)=(&queries[current_head_position_range()],&mut queriesderivatives[current_head_position_range()],&summariesderivatives[current_head_position_range()]);
						let sum=attend_to_ranges(head).zip(attention).zip(&mut *attentionderivatives).map(|((attend_to_range,a),da)|{
							let (value,valuederivatives)=(&values[attend_to_range()],&mut valuesderivatives[attend_to_range()]);
							*da=summaryvaluederivatives.iter().zip(value).zip(valuederivatives).map(|((ds,v),dv)|{
								*dv+=a*ds;
								ds*v
							}).sum();
							*da*a
						}).sum::<f32>();
						attend_to_ranges(head).zip(attention).zip(&*attentionderivatives).for_each(|((attend_to_range,a),da)|{
							let (key,keyderivatives)=(&keys[attend_to_range()],&mut keysderivatives[attend_to_range()]);
							rd_dot(key,query,keyderivatives,queryderivatives,(da-sum)*a);
						});
					});
					attentionstart=attentionstop;
				});
				input.chunks_exact(embeddingarea).zip(inputderivatives.chunks_exact_mut(embeddingarea)).zip(mask_iter().map(|(_,position)|kv_range(position))).for_each(|((input,inputderivatives),range)|{
					(rd_trans_mat_mul(inputderivatives,keyparametersderivatives,&keysderivatives[range()],0,input,embeddingarea,keyparameters),rd_trans_mat_mul(inputderivatives,valueparametersderivatives,&valuesderivatives[range()],0,input,embeddingarea,valueparameters));
				});
			});
			rd_trans_mat_mul(inputderivatives,queryparametersderivatives,queriesderivatives,0,input,embeddingarea,queryparameters);
		}else{
			*compactionheight=(*compactionheight).max(*offset+cachedtemplen+len);			//move compactionheight to the beginning of uncached temp, which is the end of cached temp
			let (input,output,parameters,temp)=self.allocate_layer(*compactionheight,len,offset,len,parameterslen,parametersposition,templen);
			let ((keyparameters,outputparameters,queryparameters,valueparameters),(keys,values,attention,queries,summaries))=(multi_split!(&*parameters,split_at,kqvparametercount,oparametercount,kqvparametercount,kqvparametercount),multi_split!(temp,split_at_mut,contextlen,contextlen,attentionlen,len,len));
			output.copy_from_slice(input);
			summaries.fill(0.0);
			trans_mat_mul(0,input,embeddingarea,queryparameters,queries);
			attention.chunks_exact_mut(attentionvolume).zip(input.chunks_exact(newvolume)).zip(keys.chunks_exact_mut(contextvolume)).zip(queries.chunks_exact(newvolume)).zip(summaries.chunks_exact_mut(newvolume)).zip(values.chunks_exact_mut(contextvolume)).for_each(|(((((attention,input),keys),queries),summaries),values)|{
				input.chunks_exact(embeddingarea).zip(mask_iter().map(|(_,position)|kv_range(position))).for_each(|(input,range)|{
					(trans_mat_mul(0,input,embeddingarea,keyparameters,&mut keys[range()]),trans_mat_mul(0,input,embeddingarea,valueparameters,&mut values[range()]));
				});
				let mut attentionoffset=0;
				mask_iter().enumerate().for_each(|(currentpositionindex,(attendto,_))|{
					let (attend_to_ranges,attentiondimension)=(|head:usize|attendto.clone().map(move|attendtoposition|head_range(head,attendtoposition)),attendto.len());
					let attentionarea=attentiondimension*headcount;
					let attention=&mut attention[attentionoffset..attentionarea+attentionoffset];
					attention.chunks_exact_mut(attentiondimension).enumerate().for_each(|(head,attention)|{
						let current_head_position_range=head_range(head,currentpositionindex);
						let (query,summaryvalue)=(&queries[current_head_position_range()],&mut summaries[current_head_position_range()]);
						let max=attend_to_ranges(head).zip(attention.iter_mut()).map(|(attend_to_range,a)|{
							*a=dot(&keys[attend_to_range()],query);
							*a
						}).fold(f32::NEG_INFINITY,f32::max);
						let scale=attention.iter_mut().map(|a|{
							*a=if *a==max{1.0}else{((*a-max)*sharpness).exp()};
							*a
						}).sum::<f32>().recip();
						attend_to_ranges(head).zip(attention.iter_mut()).for_each(|(attend_to_range,a)|{
							*a*=scale;
							let (a,value)=(*a,&values[attend_to_range()]);
							summaryvalue.iter_mut().zip(value).for_each(|(s,v)|*s+=a*v);
						});
					});
					attentionoffset+=attentionarea;
				});
			});
			trans_mat_mul(ACC|BIAS,summaries,embeddingarea,outputparameters,output);
		}
	}//TODO compactify
	/// Embeddings add to previous layer for convenient position encoding
	pub fn embed<I:Iterator<Item=usize>>(&mut self,compactionheight:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,embeddingdimension:usize,len:usize,offset:&mut usize,parametersposition:&mut usize,tokens:I,uniquetokencount:usize){
		let (parameterslen,tokenencodingranges)=(embeddingdimension*uniquetokencount,tokens.map(|t|embeddingdimension*t).map(|tx|tx..embeddingdimension+tx));
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			let ((_,inputderivatives,_,outputderivatives,_,parametersderivatives,_,_),priorderivativesoffset)=(self.allocate_layer_derivatives(compactionheight,derivatives,derivativesoffset,len,offset,len,parameterslen,parametersposition,0),*derivativesoffset);
			outputderivatives.chunks_exact(embeddingdimension).zip(tokenencodingranges).for_each(|(outputderivatives,tr)|outputderivatives.iter().zip(&mut parametersderivatives[tr]).for_each(|(d,dt)|*dt+=d));
			if *derivativesoffset<compactionheight{inputderivatives.copy_from_slice(outputderivatives)}else{*derivativesoffset=priorderivativesoffset}
		}else if *offset<compactionheight{
			let (input,output,parameters,_)=self.allocate_layer(compactionheight,len,offset,len,parameterslen,parametersposition,0);
			input.chunks_exact(embeddingdimension).zip(output.chunks_exact_mut(embeddingdimension)).zip(tokenencodingranges).for_each(|((input,output),tr)|input.iter().zip(output).zip(&parameters[tr]).for_each(|((i,o),t)|*o=i+t));
		}else{
			let (_,output,parameters,_)=self.allocate_layer(usize::MAX,0,offset,len,parameterslen,parametersposition,0);
			output.chunks_exact_mut(embeddingdimension).zip(tokenencodingranges).for_each(|(output,tr)|output.iter_mut().zip(&parameters[tr]).for_each(|(o,t)|*o+=t));
		}
	}/// Makes one from existing field data. Shorter hand than struct initialization
	pub fn from_existing_data(buffer:Vec<f32>,parameters:Vec<f32>)->Self{
		Self{buffer,parameters}
	}/// Construct from parameters. Buffer will initially have zero space
	pub fn from_parameters(parameters:Vec<f32>)->Self{Self::from_existing_data(Vec::new(),parameters)}
	/// Forces the buffer to have space at least up to the given index
	pub fn force_state(&mut self,stop:usize){
		let buffer=&mut self.buffer;
		if buffer.len()<stop{buffer.resize(stop,0.0)}
	}/// something something componentwise gelu function
	pub fn gelu_layer(&mut self,compactionheight:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,len:usize,offset:&mut usize){
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			let (input,inputderivatives,_,outputderivatives,_,_,_,_)=self.allocate_layer_derivatives(compactionheight,derivatives,derivativesoffset,len,offset,len,0,&mut 0,0);
			input.iter().zip(inputderivatives).zip(&*outputderivatives).for_each(|((&x,dx),dy)|*dx=dy*gelu_derivative(x));
		}else if *offset<compactionheight{
			let (input,output,_,_)=self.allocate_layer(compactionheight,len,offset,len,0,&mut 0,0);
			input.iter().zip(output).for_each(|(&x,y)|*y=gelu(x));
		}else{
			self.buffer[*offset..*offset+len].iter_mut().for_each(|x|*x=gelu(*x));
		}
	}/// multilayer perceptron with gelu activation function on its inner layers
	pub fn gelu_multilayer_perceptron(&mut self,bias:bool,compactionheight:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,inputdimension:usize,intermediatedimension:usize,intermediatelayercount:usize,len:&mut usize,offset:&mut usize,outputdimension:usize,parametersposition:&mut usize){
		if intermediatelayercount==0{return self.affine_layer(bias,compactionheight,derivatives,derivativesoffset,inputdimension,len,offset,outputdimension,parametersposition)}
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			self.affine_layer(bias,compactionheight,Some(derivatives),Some(derivativesoffset),intermediatedimension,len,offset,outputdimension,parametersposition);
			(0..intermediatelayercount-1).for_each(|_|{
				self.gelu_layer(compactionheight,Some(derivatives),Some(derivativesoffset),*len,offset);
				self.affine_layer(bias,compactionheight,Some(derivatives),Some(derivativesoffset),intermediatedimension,len,offset,intermediatedimension,parametersposition);
			});
			self.gelu_layer(compactionheight,Some(derivatives),Some(derivativesoffset),*len,offset);
			self.affine_layer(bias,compactionheight,Some(derivatives),Some(derivativesoffset),inputdimension,len,offset,intermediatedimension,parametersposition);
		}else{
			self.affine_layer(bias,compactionheight,None,None,inputdimension,len,offset,intermediatedimension,parametersposition);
			self.gelu_layer(compactionheight,None,None,*len,offset);
			(0..intermediatelayercount-1).for_each(|_|{
				self.affine_layer(bias,compactionheight,None,None,intermediatedimension,len,offset,intermediatedimension,parametersposition);
				self.gelu_layer(compactionheight,None,None,*len,offset);
			});
			self.affine_layer(bias,compactionheight,None,None,intermediatedimension,len,offset,outputdimension,parametersposition);
		}
	}/// gpt style transformer model function. Doesn't include position encoding, but adds embeddings to prior data, which could be made to contain position encoding
	pub fn gpt<C:Clone+ExactSizeIterator<Item=(D,usize)>,D:Clone+ExactSizeIterator<Item=usize>,I:Iterator<Item=usize>>(&mut self,bias:bool,blockcount:usize,compactionheight:&mut usize,contextdimension:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,headcount:usize,headdimension:usize,intermediateprojectiondimension:usize,maskiter:C,len:&mut usize,offset:&mut usize,parametersposition:&mut usize,tokens:I,uniquetokencount:usize){
		let (embeddingarea,mask_iter)=(headcount*headdimension,||maskiter.clone());
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			self.affine_layer(bias,*compactionheight,Some(derivatives),Some(derivativesoffset),embeddingarea,len,offset,uniquetokencount,parametersposition);
			self.normalization_layer(*compactionheight,Some(derivatives),Some(derivativesoffset),embeddingarea,*len,offset,parametersposition);
			(0..blockcount).for_each(|_|self.transformer_block(bias,compactionheight,contextdimension,Some(derivatives),Some(derivativesoffset),headcount,headdimension,intermediateprojectiondimension,mask_iter(),*len,offset,parametersposition));
			self.embed(*compactionheight,Some(derivatives),Some(derivativesoffset),embeddingarea,*len,offset,parametersposition,tokens,uniquetokencount);
		}else{
			self.embed(*compactionheight,None,None,embeddingarea,*len,offset,parametersposition,tokens,uniquetokencount);
			(0..blockcount).for_each(|_|self.transformer_block(bias,compactionheight,contextdimension,None,None,headcount,headdimension,intermediateprojectiondimension,mask_iter(),*len,offset,parametersposition));
			self.normalization_layer(*compactionheight,None,None,embeddingarea,*len,offset,parametersposition);
			self.affine_layer(bias,*compactionheight,None,None,embeddingarea,len,offset,uniquetokencount,parametersposition);
		}
	}/// Stochastic gradient descent with momentum. Should be applied on the adjustment step after differentiation
	pub fn msgd(&mut self,derivatives:&mut Self,momentum:f32,rate:f32){
		derivatives.parameters.iter_mut().zip(&mut self.parameters).for_each(|(d,x)|{
			*x+=*d*rate;
			*d*=momentum;
		});
	}/// layer norm!~
	pub fn normalization_layer(&mut self,compactionheight:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,dimension:usize,len:usize,offset:&mut usize,parametersposition:&mut usize){
		const EPSILON:f32=1.0E-5;
		let (arl,li,parameterslen)=(len/dimension,(dimension as f32).recip(),dimension*2);
		let average_and_reciprocal_deviation=|data:&[f32]|{
			let sum=data.iter().sum::<f32>();
			let average=li*sum;
			(average,(data.iter().map(|v|average-v).map(|v|v*v).sum::<f32>()*li+EPSILON).recip().sqrt())
		};
		let templen=arl*2;
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			let (input,inputderivatives,_,outputderivatives,parameters,parametersderivatives,temp,_)=self.allocate_layer_derivatives(compactionheight,derivatives,derivativesoffset,len,offset,len,parameterslen,parametersposition,templen);
			let ((averages,reciprocaldeviations),(_,gamma),(betaderivatives,gammaderivatives))=(temp.split_at(arl),parameters.split_at(dimension),parametersderivatives.split_at_mut(dimension));
			averages.iter().zip(inputderivatives.chunks_exact_mut(dimension)).zip(outputderivatives.chunks_exact(dimension)).zip(reciprocaldeviations).zip(input.chunks_exact(dimension)).for_each(|((((a,dx),dy),r),x)|{
				let r2li=r*r*li;
				let ar2li=a*r2li;
				let (dygysum,dygyxsum)=dy.iter().zip(gamma).map(|(dy,g)|dy*g).zip(x).map(|(dyg,x)|(dyg,dyg*x)).fold((0.0,0.0),|(a,b),(x,y)|(a+x,b+y));
				let (ar2lidygysum,dygyxsumr2li)=(ar2li*dygysum,dygyxsum*r2li);
				let dygysumli=dygysum*li;
				betaderivatives.iter_mut().zip(&mut *gammaderivatives).zip(dx.iter_mut()).zip(dy).zip(gamma).zip(x).for_each(|(((((db,dg),dx),dy),g),x)|{
					let xnma=x-a;
					*db+=dy;
					*dg+=dy*xnma*r;
					*dx=((ar2lidygysum-dygyxsumr2li)*xnma-dygysumli+dy*g)*r;
				});
			});
		}else if *offset<compactionheight{
			let (input,output,parameters,temp)=self.allocate_layer(compactionheight,len,offset,len,parameterslen,parametersposition,templen);
			let ((averages,reciprocaldeviations),(beta,gamma))=(temp.split_at_mut(arl),parameters.split_at(dimension));
			averages.iter_mut().zip(reciprocaldeviations).zip(input.chunks_exact(dimension)).zip(output.chunks_exact_mut(dimension)).for_each(|(((average,reciprocaldeviation),x),y)|{
				let (a,r)=average_and_reciprocal_deviation(x);
				(*average,*reciprocaldeviation)=(a,r);
				beta.iter().zip(gamma).zip(x).zip(y).for_each(|(((b,g),x),y)|*y=(x-a)*g*r+b);
			});
		}else{
			let (_,output,parameters,_)=self.allocate_layer(usize::MAX,0,offset,len,parameterslen,parametersposition,0);
			let (beta,gamma)=parameters.split_at(dimension);
			output.chunks_exact_mut(dimension).for_each(|data|{
				let (a,r)=average_and_reciprocal_deviation(data);
				beta.iter().zip(data).zip(gamma).for_each(|((b,v),g)|*v=(*v-a)*g*r+b);
			});
		}
	}/// The sort of projection layer used in a transformer architecture
	pub fn projection_layer(&mut self,bias:bool,compactionheight:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,embeddingdimension:usize,intermediatedimension:usize,len:usize,offset:&mut usize,parametersposition:&mut usize){
		let biasflag=BIAS*bias as u8;
		let (compacttemplen,inputparameterslen,outputparameterslen)=(len/embeddingdimension*intermediatedimension,(bias as usize+embeddingdimension)*intermediatedimension,(bias as usize+intermediatedimension)*embeddingdimension);
		let (parameterslen,uncompacttemplen)=(inputparameterslen+outputparameterslen,compacttemplen*2);
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			let (input,inputderivatives,_,outputderivatives,parameters,parametersderivatives,temp,tempderivatives)=self.allocate_layer_derivatives(compactionheight,derivatives,derivativesoffset,len,offset,len,parameterslen,parametersposition,uncompacttemplen);
			let ((inputparameters,outputparameters),(inputparametersderivatives,outputparametersderivatives),(tempa,tempb),(tempaderivatives,tempbderivatives))=(parameters.split_at(inputparameterslen),parametersderivatives.split_at_mut(inputparameterslen),temp.split_at(compacttemplen),tempderivatives.split_at_mut(compacttemplen));
			inputderivatives.copy_from_slice(outputderivatives);
			tempbderivatives.fill(0.0);
			rd_trans_mat_mul(tempbderivatives,outputparametersderivatives,outputderivatives,ACC|biasflag,tempb,intermediatedimension,outputparameters);
			tempa.iter().zip(&mut *tempaderivatives).zip(&*tempbderivatives).for_each(|((&x,dx),dy)|*dx=dy*gelu_derivative(x));
			rd_trans_mat_mul(inputderivatives,inputparametersderivatives,tempaderivatives,biasflag,input,embeddingdimension,inputparameters);
		}else if *offset+len<compactionheight{//TODO could probably also just reuse input
			let (input,output,parameters,temp)=self.allocate_layer(compactionheight,len,offset,len,parameterslen,parametersposition,uncompacttemplen);
			let ((inputparameters,outputparameters),(tempa,tempb))=(parameters.split_at(inputparameterslen),temp.split_at_mut(compacttemplen));
			output.copy_from_slice(input);
			trans_mat_mul(biasflag,input,embeddingdimension,inputparameters,tempa);
			tempa.iter().zip(&mut *tempb).for_each(|(&x,y)|*y=gelu(x));
			trans_mat_mul(ACC|biasflag,tempb,intermediatedimension,outputparameters,output);
		}else{
			let (input,output,parameters,temp)=self.allocate_layer(compactionheight,len,offset,len,parameterslen,parametersposition,compacttemplen);
			let (inputparameters,outputparameters)=parameters.split_at_mut(inputparameterslen);
			output.copy_from_slice(input);
			trans_mat_mul(biasflag,input,embeddingdimension,inputparameters,temp);
			temp.iter_mut().for_each(|x|*x=gelu(*x));
			trans_mat_mul(ACC|biasflag,temp,intermediatedimension,outputparameters,output);
		}
	}/// rprop optimization step
	pub fn rprop(&mut self,derivatives:&mut Self,hm:f32,hp:f32){
		let (parameters,parametersderivatives)=(&mut self.parameters,&mut derivatives.parameters);
		let parameterslen=parameters.len();
		parametersderivatives.resize(parameterslen*2,0.0);
		let (derivatives,directiondata)=parametersderivatives.split_at_mut(parameterslen);
		derivatives.iter_mut().zip(directiondata).zip(parameters).for_each(|((d,directiondata),x)|{
			let (derivativesign,priorderivativesign)=(d.is_sign_negative(),directiondata.is_sign_negative());
			*directiondata*=if derivativesign==priorderivativesign{hp}else{-hm};
			if directiondata.is_nan()||*directiondata==0.0{*directiondata=*d}
			*d=0.0;
			*directiondata=directiondata.clamp(-1.0,1.0);
			*x+=*directiondata;
		});
	}/// Retrieves a layer with the length and offset
	pub fn state(&self,len:usize,offset:usize)->&[f32]{&self.buffer[offset..len+offset]}
	/// Retrieves a layer with the length and offset, but mutably
	pub fn state_mut(&mut self,len:usize,offset:usize)->&mut [f32]{&mut self.buffer[offset..len+offset]}
	/// A transformer block consisting of layer norm, attention, layer norm, projection
	pub fn transformer_block<C:Clone+ExactSizeIterator<Item=(D,usize)>,D:Clone+ExactSizeIterator<Item=usize>>(&mut self,bias:bool,compactionheight:&mut usize,contextdimension:usize,derivatives:Option<&mut Self>,derivativesoffset:Option<&mut usize>,headcount:usize,headdimension:usize,intermediateprojectiondimension:usize,maskiter:C,len:usize,offset:&mut usize,parametersposition:&mut usize){
		let embeddingarea=headcount*headdimension;
		if let (Some(derivatives),Some(derivativesoffset))=(derivatives,derivativesoffset){
			self.projection_layer(bias,*compactionheight,Some(derivatives),Some(derivativesoffset),embeddingarea,intermediateprojectiondimension,len,offset,parametersposition);
			self.normalization_layer(*compactionheight,Some(derivatives),Some(derivativesoffset),embeddingarea,len,offset,parametersposition);
			self.attention_layer(compactionheight,contextdimension,Some(derivatives),Some(derivativesoffset),headcount,headdimension,len,maskiter,offset,parametersposition);
			self.normalization_layer(*compactionheight,Some(derivatives),Some(derivativesoffset),embeddingarea,len,offset,parametersposition);
		}else{
			self.normalization_layer(*compactionheight,None,None,embeddingarea,len,offset,parametersposition);
			self.attention_layer(compactionheight,contextdimension,None,None,headcount,headdimension,len,maskiter,offset,parametersposition);
			self.normalization_layer(*compactionheight,None,None,embeddingarea,len,offset,parametersposition);
			self.projection_layer(bias,*compactionheight,None,None,embeddingarea,intermediateprojectiondimension,len,offset,parametersposition);
		}
	}
}///mat mul flag to accumulate into the output rather than setting it
pub const ACC:u8=1;
///mat mul flag to add bias in the same operation
pub const BIAS:u8=2;
///dot product of two vectors represented by f32 slices
pub fn dot(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(a,b)|a*b).sum()}
///cross entropy in nats of a softmax of logits with a 'one hot' probability distribution. (a distribution where one value is 1 and the rest are 0). The input 'correct' gives the index of the 1.
pub fn entropic_one_hot_error(correct:usize,logits:&[f32],sharpness:f32)->f32{
	let lic=||logits.iter().copied();
	let m=if sharpness<0.0{lic().fold(f32::INFINITY,f32::min)}else{lic().fold(f32::NEG_INFINITY,f32::max)};
	let lnchances=||lic().map(|x|if m==x{0.0}else{(x-m)*sharpness});
	let lnsum=lnchances().map(f32::exp).sum::<f32>().ln();
	lnsum-lnchances().nth(correct).expect("correct bit is out of range")
}///approximation of the Gaussian Error Linear Unit. Apparently its a good activation function or something idk
pub fn gelu(x:f32)->f32{(((x*x*x*0.044715+x)*0.79788456).tanh()+1.0)*x*0.5}//TODO magic numbers
/// gelu derivative
pub fn gelu_derivative(x:f32)->f32{
	let x2=x*x;
	let t=(x2*x*0.044715+x)*0.79788456;//TODO magic numbers
	let b=t.cosh().recip();
	((x2*0.134145+1.0)*b*b*x*0.79788456+t.tanh()+1.0)*0.5
}///position encodings. TODO think of better doc comments for this and in general
pub fn position_encodings(embeddingdimension:usize)->impl Clone+Iterator<Item=f32>{
	let m=-2.0/(embeddingdimension as f32);
	(0..).map(move|x|{
		let y=x%embeddingdimension;
		let x=(x-y)as f32;
		let z=x*10000f32.powf((y/2)as f32*m);
		if y&1==0{z.sin()}else{z.cos()}
	})
}///accumulate the derivatives of loss with the dot product of two vectors, represented by slices. 
pub fn rd_dot(a:&[f32],b:&[f32],da:&mut [f32],db:&mut [f32],dc:f32){
	a.iter().zip(b).zip(da).zip(db).for_each(|(((a,b),da),db)|{
		*da+=b*dc;
		*db+=a*dc;
	});
}///compute the derivatives of loss with the cross entropy of the softmax of logits with a 'one hot' distribution. does not accumulate
pub fn rd_entropic_one_hot_error(correct:usize,errorderivative:f32,logits:&[f32],logitsderivatives:&mut [f32],sharpness:f32){//TODO If we don't want to make logitsderivatives acc, performance could be improved without too much effort byt using logitsderivatives as scratch space rather than recomputing chances
	let lic=||logits.iter().copied();
	let m=if sharpness<0.0{lic().fold(f32::INFINITY,f32::min)}else{lic().fold(f32::NEG_INFINITY,f32::max)};
	let chances=||lic().map(|x|if m==x{1.0}else{((x-m)*sharpness).exp()});
	let r=chances().sum::<f32>().recip()*errorderivative*sharpness;
	chances().zip(&mut *logitsderivatives).for_each(|(chance,dl)|*dl=chance*r);
	logitsderivatives[correct]-=errorderivative*sharpness;
}///compute the derivatives of loss with the squared error of logits with a target logit distribution. does not accumulate
pub fn rd_squared_error(errorderivative:f32,logits:&[f32],logitsderivatives:&mut [f32],target:&[f32]){logits.iter().zip(logitsderivatives).zip(target).for_each(|((l,d),t)|*d=(l-t)*errorderivative*2.0)}
///accumulate the derivatives of loss with x and y given the derivatives of loss with the z=x(yT) computed by trans_mat_mul.
pub fn rd_trans_mat_mul(dx:&mut [f32],dy:&mut [f32],dz:&[f32],flags:u8,x:&[f32],xdimension:usize,y:&[f32]){
	let bias=BIAS&flags!=0;
	let ydimension=bias as usize+xdimension;
	let zdimension=y.len()/ydimension;
	dx.chunks_exact_mut(xdimension).zip(x.chunks_exact(xdimension)).zip(dz.chunks_exact(zdimension)).for_each(|((dx,x),dz)|dy.chunks_exact_mut(ydimension).zip(dz).zip(y.chunks_exact(ydimension)).for_each(|((dy,&dz),y)|{
		if bias{dy[0]+=dz}													//z[n][k]=sumi(x[n][i]*y[k][i])		 z[a][b]=sumc( x[a][c]* y[b][c])
		rd_dot(x,&y[bias as usize..],dx,&mut dy[bias as usize..],dz);		//dx[n][i]=sumk(dz[n][k]*y[k][i])	dx[a][b]=sumc(dz[a][c]* y[c][b])
	}));																	//dy[k][i]=sumn(dz[n][k]*x[n][i])	dy[a][b]=sumc(dz[c][a]* x[c][b])
}///pseudorandomly updates a 16 byte seed and returns a 4 byte float evenly distributed on [-1.0, 1.0)
pub fn rfloat(seed:&mut u128)->f32{
	update_seed(seed);
	*seed as i32 as f32/(0x80000000 as u32 as f32)
}///pseudorandomly updates a 16 byte seed and returns a usize evenly distributed on [0, bound)
pub fn rsize(bound:usize,seed:&mut u128)->usize{
	assert!(bound!=0);
	let mask=bound.next_power_of_two()-1;
	loop{
		update_seed(seed);
		let result=(*seed as usize)&mask;
		if bound>result{break result}
	}
}///creates a 16 byte seed from the current time
pub fn rseed()->u128{SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()}
///pseudorandomly chooses an index from the probability distribution given by the softmax of logits
pub fn soft_choose(logits:&[f32],seed:&mut u128,sharpness:f32)->usize{//TODO could be faster with a temp slice or just mutable logits if we aren't reusing them
	let lic=||logits.iter().copied();
	let m=if sharpness<0.0{lic().fold(f32::INFINITY,f32::min)}else{lic().fold(f32::NEG_INFINITY,f32::max)};
	let chances=||lic().map(|x|if m==x{1.0}else{((x-m)*sharpness).exp()});
	let mut choice=(rfloat(seed)*0.5+1.0)*chances().sum::<f32>();
	for (n,chance)in chances().enumerate(){
		choice-=chance;
		if choice<0.0{return n}
	}0
}///computes the sum of squared differences between the logits and the target logit distribution
pub fn squared_error(logits:&[f32],target:&[f32])->f32{logits.iter().zip(target).map(|(l,t)|l-t).map(|x|x*x).sum()}
///multiplies matrix x with column height xdimension by y, or rather y's transpose since that is more convenient for the computer and saves a transposition step in some cases since nothing in this code wants to do a normal mat mul. The acc flag will make it accumulate into z rather than setting it. The bias flag will make bias in the neural network convenient by doing an affine transform interpreting x as a coordinate matrix and y as an affine transformation matrix, with the bottom row (or right column of y since its transposed) implicit.
pub fn trans_mat_mul(flags:u8,x:&[f32],xdimension:usize,y:&[f32],z:&mut [f32]){
	let (acc,bias)=(ACC&flags!=0,BIAS&flags!=0);
	let ydimension=bias as usize+xdimension;
	let zdimension=y.len()/ydimension;
	x.chunks_exact(xdimension).zip(z.chunks_exact_mut(zdimension)).for_each(|(x,z)|y.chunks_exact(ydimension).zip(z).for_each(|(y,z)|*z=dot(&y[bias as usize..],x)+if acc{*z}else{0.0}+if bias{y[0]}else{0.0}));
}///updates a 16 byte seed, currently by using an xorshift algorithm
pub fn update_seed(seed:&mut u128){
	*seed^=*seed<<15;
	*seed^=*seed>>4;
	*seed^=*seed<<21;
}
